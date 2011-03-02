/*
 *      Linear-chain CRF training.
 *
 * Copyright (c) 2007-2010, Naoaki Okazaki
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the names of the authors nor the names of its contributors
 *       may be used to endorse or promote products derived from this
 *       software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* $Id: crfvo_learn.c 176 2010-07-14 09:31:04Z naoaki $ */

#ifdef    HAVE_CONFIG_H
#include <config.h>
#endif/*HAVE_CONFIG_H*/

#include <os.h>

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>

#include <crfsuite.h>
#include "params.h"
#include "mt19937ar.h"

#include "logging.h"
#include "crfvo.h"

#define    FEATURE(trainer, k) \
    (&(trainer)->features[(k)])
#define    ATTRIBUTE(trainer, a) \
    (&(trainer)->attributes[(a)])

void crfvol_enum_features(crfvol_t* trainer, const crf_sequence_t* seq, update_feature_t func, double* logp)
{
    crfvo_context_t* ctx = trainer->ctx;
    const int T = seq->num_items;
    const int L = trainer->num_labels;
	int i, j, t;

	for (t = 0; t < T; ++t) {
		crfvo_path_score_t* path_scores = ctx->path_scores[t];
		int* fids = ctx->fids_refs[t];
		int n = ctx->num_paths[t];
		int fid_counter = 0;
		for (i = 0; i < n; ++i) {
			int fid_num = path_scores[i].path.feature_count;
			for (j = 0; j < fid_num; ++j) {
				floatval_t prob = path_scores[i].score;
				int fid = fids[fid_counter];
				crfvol_feature_t* f = FEATURE(trainer, fid);
				fid_counter++;
				func(f, fid, prob, 1.0, trainer, seq, t);				
			}
		}
	}
}

static int init_feature_references(crfvol_t* trainer, const int A, const int L)
{
    int i, k;
    feature_refs_t *fl = NULL;
    const int K = trainer->num_features;
    const crfvol_feature_t* features = trainer->features;

    /*
        The purpose of this routine is to collect references (indices) of:
        - features fired by each attribute (trainer->attributes)
    */

    /* Initialize. */
    trainer->attributes = NULL;

    /* Allocate arrays for feature references. */
    trainer->attributes = (feature_refs_t*)calloc(A, sizeof(feature_refs_t));
    if (trainer->attributes == NULL) goto error_exit;

    /*
        Firstly, loop over the features to count the number of references.
        We don't want to use realloc() to avoid memory fragmentation.
     */
    for (k = 0;k < K;++k) {
        const crfvol_feature_t *f = &features[k];
        trainer->attributes[f->attr].num_features++;
    }

    /*
        Secondarily, allocate memory blocks to store the feature references.
        We also clear fl->num_features fields, which will be used to indicate
        the offset positions in the last phase.
     */
    for (i = 0;i < trainer->num_attributes;++i) {
        fl = &trainer->attributes[i];
        fl->fids = (int*)calloc(fl->num_features, sizeof(int));
        if (fl->fids == NULL) goto error_exit;
        fl->num_features = 0;
    }

    /*
        At last, store the feature indices.
     */
    for (k = 0;k < K;++k) {
        const crfvol_feature_t *f = &features[k];
        fl = &trainer->attributes[f->attr];
        fl->fids[fl->num_features++] = k;
    }

    return 0;

error_exit:
    if (trainer->attributes == NULL) {
        for (i = 0;i < A;++i) free(trainer->attributes[i].fids);
        free(trainer->attributes);
        trainer->attributes = NULL;
    }
    return -1;
}

int crfvol_prepare(
    crfvol_t* trainer,
    int num_labels,
    int num_attributes,
    int max_item_length,
	int max_paths,
    crfvol_features_t* features
    )
{
    int ret = 0;
    const int L = num_labels;
    const int A = num_attributes;
    const int T = max_item_length;

    /* Set basic parameters. */
    trainer->num_labels = L;
    trainer->num_attributes = A;

    /* Construct a CRF context. */
    trainer->ctx = crfvoc_new(L, T, max_paths);
    if (trainer->ctx == NULL) {
        ret = CRFERR_OUTOFMEMORY;
        goto error_exit;
    }

    /* Initialization for features and their weights. */
    trainer->features = features->features;
    trainer->num_features = features->num_features;
    trainer->w = (floatval_t*)calloc(trainer->num_features, sizeof(floatval_t));
    if (trainer->w == NULL) {
        ret = CRFERR_OUTOFMEMORY;
        goto error_exit;
    }

    /* Allocate the work space for probability calculation. */
    trainer->prob = (floatval_t*)calloc(L, sizeof(floatval_t));
    if (trainer->prob == NULL) {
        ret = CRFERR_OUTOFMEMORY;
        goto error_exit;
    }

    /* Initialize the feature references. */
    init_feature_references(trainer, A, L);

	return ret;

error_exit:
    free(trainer->attributes);
    free(trainer->prob);
    free(trainer->ctx);
    return 0;
}

void crfvol_preprocess(
    crfvol_t* trainer
	)
{
	int i;
	logging(trainer->lg, "Compiling...\n");
	logging_progress_start(trainer->lg);

	for (i = 0; i < trainer->num_sequences; ++i) {
		crfvopp_preprocess_sequence(
			(crfvopp_t*)trainer->preprocessor,
			trainer->attributes,
			trainer->features,
			trainer->num_labels,
			&trainer->seqs[i]);

		if (trainer->max_paths < trainer->seqs[i].max_paths) {
			trainer->max_paths = trainer->seqs[i].max_paths;
		}
		logging_progress(trainer->lg, (100 * i) / trainer->num_sequences);
	}

	logging_progress_end(trainer->lg);
}

int crfvol_set_feature_freqs(
	crfvol_t* trainer,
	crfvol_features_t* features
	)
{
	int i, t, n, ret;
	int* feature_freqs = (int*)calloc(trainer->num_features, sizeof(int));
	int* feature_last_indexes = (int*)malloc(trainer->max_paths * sizeof(int));

	for (i = 0; i < trainer->num_sequences; ++i) {
		crf_sequence_t* seq = &trainer->seqs[i];
		for (t = 0; t < seq->num_items; ++t) {
			crf_item_t* item = &seq->items[t];
			crfvopd_t* preprocessed_data = (crfvopd_t*)item->preprocessed_data;
			int feature_last_index = 0;
			for (n = 0; n < preprocessed_data->num_paths; ++n) {
				feature_last_index += preprocessed_data->paths[n].feature_count;
				feature_last_indexes[n] = feature_last_index;
			}
			n = preprocessed_data->training_path_index;
			while (n > 0) {
				int j;
				for (j = feature_last_indexes[n-1]; j < feature_last_indexes[n]; ++j) {
					feature_freqs[preprocessed_data->fids[j]]++;
				}
				n = preprocessed_data->paths[n].longest_suffix_index;
			}
		}
	}
	ret = 1;
	for (i = 0; i < trainer->num_features; ++i) {
		if (trainer->features[i].freq != feature_freqs[i]) {
			trainer->features[i].freq = feature_freqs[i];
		}
	}
	free(feature_freqs);
	free(feature_last_indexes);
	return ret;
}

static int crfvol_exchange_options(crf_params_t* params, crfvol_option_t* opt, int mode)
{
    BEGIN_PARAM_MAP(params, mode)
        DDX_PARAM_STRING(
            "algorithm", opt->algorithm, "lbfgs",
            "The training algorithm."
            )
        DDX_PARAM_FLOAT(
            "feature.minfreq", opt->feature_minfreq, 0.0,
            "The minimum frequency of features."
            )
    END_PARAM_MAP()

    crfvol_lbfgs_options(params, opt, mode);

    return 0;
}

void crfvol_shuffle(int *perm, int N, int init)
{
    int i, j, tmp;

    if (init) {
        /* Initialize the permutation if necessary. */
        for (i = 0;i < N;++i) {
            perm[i] = i;
        }
    }

    for (i = 0;i < N;++i) {
        j = mt_genrand_int31() % N;
        tmp = perm[j];
        perm[j] = perm[i];
        perm[i] = tmp;
    }
}

crfvol_t* crfvol_new()
{
    crfvol_t* trainer = (crfvol_t*)calloc(1, sizeof(crfvol_t));
    trainer->lg = (logging_t*)calloc(1, sizeof(logging_t));
	trainer->exp_weight = 0;
	trainer->preprocessor = 0;

    /* Create an instance for CRF parameters. */
    trainer->params = params_create_instance();
    /* Set the default parameters. */
    crfvol_exchange_options(trainer->params, &trainer->opt, 0);

    return trainer;
}

void crfvol_delete(crfvol_t* trainer)
{
    if (trainer != NULL) {
        free(trainer->lg);
		free(trainer->exp_weight);
		if (trainer->preprocessor_delete_func) {
			trainer->preprocessor_delete_func(trainer->preprocessor);
		}
		free(trainer->preprocessor);
    }
	free(trainer);
}

int crf_train_tag(crf_tagger_t* tagger, crf_sequence_t *inst, crf_output_t* output)
{
    int i;
    floatval_t logscore = 0;
    crfvol_t *crfvot = (crfvol_t*)tagger->internal;
    const floatval_t* exp_weight = crfvot->exp_weight;
    const int K = crfvot->num_features;
    crfvo_context_t* ctx = crfvot->ctx;
	int max_path = 0;

	for (i = 0; i < inst->num_items; ++i) {
		if (inst->items[i].preprocessed_data == 0) {
			crfvopp_preprocess_sequence(
				(crfvopp_t*)crfvot->preprocessor,
				crfvot->attributes,
				crfvot->features,
				crfvot->num_labels,
				inst);
			break;
		}
	}
    crfvoc_set_num_items(ctx, inst->num_items, inst->max_paths);

    crfvoc_set_context(ctx, inst);
	crfvoc_set_weight(ctx, exp_weight);
    logscore = crfvoc_decode(crfvot->ctx);

    crf_output_init_n(output, inst->num_items);
    output->probability = logscore;
    for (i = 0;i < inst->num_items;++i) {
        output->labels[i] = crfvot->ctx->labels[i];
    }
    output->num_labels = inst->num_items;

    return 0;
}

void crf_train_set_message_callback(crf_trainer_t* trainer, void *instance, crf_logging_callback cbm)
{
    crfvol_t *crfvot = (crfvol_t*)trainer->internal;
    crfvot->lg->func = cbm;
    crfvot->lg->instance = instance;
}

void crf_train_set_evaluate_callback(crf_trainer_t* trainer, void *instance, crf_evaluate_callback cbe)
{
    crfvol_t *crfvot = (crfvol_t*)trainer->internal;
    crfvot->cbe_instance = instance;
    crfvot->cbe_proc = cbe;
}

static int crf_train_train(
    crf_trainer_t* trainer,
    void* instances,
    int num_instances,
    int num_labels,
    int num_attributes
    )
{
    int i, max_item_length;
    int ret = 0;
    floatval_t sigma = 10, *best_w = NULL;
    crf_sequence_t* seqs = (crf_sequence_t*)instances;
    crfvol_features_t* features = NULL;
    crfvol_t *crfvot = (crfvol_t*)trainer->internal;
    crf_params_t *params = crfvot->params;
    crfvol_option_t *opt = &crfvot->opt;

    /* Obtain the maximum number of items. */
    max_item_length = 0;
    for (i = 0;i < num_instances;++i) {
        if (max_item_length < seqs[i].num_items) {
            max_item_length = seqs[i].num_items;
        }
    }

    /* Access parameters. */
    crfvol_exchange_options(crfvot->params, opt, -1);

    /* Report the parameters. */
    logging(crfvot->lg, "Training first-order linear-chain CRFs (trainer.crfvo)\n");
    logging(crfvot->lg, "\n");

    /* Generate features. */
    logging(crfvot->lg, "Feature generation\n");
    logging(crfvot->lg, "feature.minfreq: %f\n", opt->feature_minfreq);
    crfvot->clk_begin = clock();
    features = crfvol_generate_features(
        seqs,
        num_instances,
        num_labels,
        num_attributes,
        opt->feature_minfreq,
        crfvot->lg->func,
        crfvot->lg->instance
        );
    logging(crfvot->lg, "Number of features: %d\n", features->num_features);
    logging(crfvot->lg, "Seconds required: %.3f\n", (clock() - crfvot->clk_begin) / (double)CLOCKS_PER_SEC);
    logging(crfvot->lg, "\n");

    /* Preparation for training. */
	crfvol_prepare(crfvot, num_labels, num_attributes, max_item_length, 0, features);
    crfvot->num_attributes = num_attributes;
    crfvot->num_labels = num_labels;
    crfvot->num_sequences = num_instances;
    crfvot->seqs = seqs;

	crfvot->preprocessor = malloc(sizeof(crfvopp_t));
	crfvot->preprocessor_delete_func = (void (*)(void*))crfvopp_delete;
	crfvopp_new((crfvopp_t*)crfvot->preprocessor);

	/* preprocess */
	crfvol_preprocess(crfvot);
	crfvol_set_feature_freqs(crfvot, features);

	crfvoc_set_num_items(crfvot->ctx, max_item_length, crfvot->max_paths);

    crfvot->tagger.internal = crfvot;
    crfvot->tagger.tag = crf_train_tag;

    if (strcmp(opt->algorithm, "lbfgs") == 0) {
        ret = crfvol_lbfgs(crfvot, opt);
    } else {
        return CRFERR_INTERNAL_LOGIC;
    }

    return ret;
}

/*#define    CRF_TRAIN_SAVE_NO_PRUNING    1*/

static int crf_train_save(crf_trainer_t* trainer, const char *filename, crf_dictionary_t* attrs, crf_dictionary_t* labels)
{
    crfvol_t *crfvot = (crfvol_t*)trainer->internal;
    int a, k, l, ret;
    int *fmap = NULL, *amap = NULL;
    crfvomw_t* writer = NULL;
    const feature_refs_t *edge = NULL, *attr = NULL;
    const floatval_t *w = crfvot->w;
    const floatval_t threshold = 0.01;
    const int L = crfvot->num_labels;
    const int A = crfvot->num_attributes;
    const int K = crfvot->num_features;
    int J = 0, B = 0;

    /* Start storing the model. */
    logging(crfvot->lg, "Storing the model\n");
    crfvot->clk_begin = clock();

    /* Allocate and initialize the feature mapping. */
    fmap = (int*)calloc(K, sizeof(int));
    if (fmap == NULL) {
        goto error_exit;
    }
#ifdef    CRF_TRAIN_SAVE_NO_PRUNING
    for (k = 0;k < K;++k) fmap[k] = k;
    J = K;
#else
    for (k = 0;k < K;++k) fmap[k] = -1;
#endif/*CRF_TRAIN_SAVE_NO_PRUNING*/

    /* Allocate and initialize the attribute mapping. */
    amap = (int*)calloc(A, sizeof(int));
    if (amap == NULL) {
        goto error_exit;
    }
#ifdef    CRF_TRAIN_SAVE_NO_PRUNING
    for (a = 0;a < A;++a) amap[a] = a;
    B = A;
#else
    for (a = 0;a < A;++a) amap[a] = -1;
#endif/*CRF_TRAIN_SAVE_NO_PRUNING*/

    /*
     *    Open a model writer.
     */
    writer = crfvomw(filename);
    if (writer == NULL) {
        goto error_exit;
    }

    /* Open a feature chunk in the model file. */
    if (ret = crfvomw_open_features(writer)) {
        goto error_exit;
    }

    /* Determine a set of active features and attributes. */
    for (k = 0;k < crfvot->num_features;++k) {
        crfvol_feature_t* f = &crfvot->features[k];
        if (w[k] != 0) {
            int attr;
            crfvom_feature_t feat;

#ifndef    CRF_TRAIN_SAVE_NO_PRUNING
            /* The feature (#k) will have a new feature id (#J). */
            fmap[k] = J++;        /* Feature #k -> #fmap[k]. */

            /* Map the source of the field. */
            /* The attribute #(f->src) will have a new attribute id (#B). */
            if (amap[f->attr] < 0) amap[f->attr] = B++;    /* Attribute #a -> #amap[a]. */
            attr = amap[f->attr];
#endif/*CRF_TRAIN_SAVE_NO_PRUNING*/

            feat.order = f->order;
            feat.attr = attr;
            memcpy(feat.label_sequence, f->label_sequence, sizeof(feat.label_sequence[0]) * MAX_ORDER);
            feat.weight = w[k];

            /* Write the feature. */
            if (ret = crfvomw_put_feature(writer, fmap[k], &feat)) {
                goto error_exit;
            }
        }
    }

    /* Close the feature chunk. */
    if (ret = crfvomw_close_features(writer)) {
        goto error_exit;
    }

    logging(crfvot->lg, "Number of active features: %d (%d)\n", J, K);
    logging(crfvot->lg, "Number of active attributes: %d (%d)\n", B, A);
    logging(crfvot->lg, "Number of active labels: %d (%d)\n", L, L);

    /* Write labels. */
    logging(crfvot->lg, "Writing labels\n", L);
    if (ret = crfvomw_open_labels(writer, L)) {
        goto error_exit;
    }
    for (l = 0;l < L;++l) {
        const char *str = NULL;
        labels->to_string(labels, l, &str);
        if (str != NULL) {
            if (ret = crfvomw_put_label(writer, l, str)) {
                goto error_exit;
            }
            labels->free_(labels, str);
        }
    }
    if (ret = crfvomw_close_labels(writer)) {
        goto error_exit;
    }

    /* Write attributes. */
    logging(crfvot->lg, "Writing attributes\n");
    if (ret = crfvomw_open_attrs(writer, B)) {
        goto error_exit;
    }
    for (a = 0;a < A;++a) {
        if (0 <= amap[a]) {
            const char *str = NULL;
            attrs->to_string(attrs, a, &str);
            if (str != NULL) {
                if (ret = crfvomw_put_attr(writer, amap[a], str)) {
                    goto error_exit;
                }
                attrs->free_(attrs, str);
            }
        }
    }
    if (ret = crfvomw_close_attrs(writer)) {
        goto error_exit;
    }

    /* Write attribute feature references. */
    logging(crfvot->lg, "Writing feature references for attributes\n");
    if (ret = crfvomw_open_attrrefs(writer, B)) {
        goto error_exit;
    }
    for (a = 0;a < A;++a) {
        if (0 <= amap[a]) {
            attr = ATTRIBUTE(crfvot, a);
            if (ret = crfvomw_put_attrref(writer, amap[a], attr, fmap)) {
                goto error_exit;
            }
        }
    }
    if (ret = crfvomw_close_attrrefs(writer)) {
        goto error_exit;
    }

    /* Close the writer. */
    crfvomw_close(writer);
    logging(crfvot->lg, "Seconds required: %.3f\n", (clock() - crfvot->clk_begin) / (double)CLOCKS_PER_SEC);
    logging(crfvot->lg, "\n");

    free(amap);
    free(fmap);
    return 0;

error_exit:
    if (writer != NULL) {
        crfvomw_close(writer);
    }
    if (amap != NULL) {
        free(amap);
    }
    if (fmap != NULL) {
        free(fmap);
    }
    return ret;
}

static int crf_train_addref(crf_trainer_t* trainer)
{
    return crf_interlocked_increment(&trainer->nref);
}

static int crf_train_release(crf_trainer_t* trainer)
{
    int count = crf_interlocked_decrement(&trainer->nref);
    if (count == 0) {
		crfvol_t* crfvot = (crfvol_t*)trainer->internal;
		if (crfvot->preprocessor) {
			crfvot->preprocessor_delete_func(crfvot->preprocessor);
			free(crfvot->preprocessor);
		}
    }
    return count;
}

static crf_params_t* crf_train_params(crf_trainer_t* trainer)
{
    crfvol_t *crfvot = (crfvol_t*)trainer->internal;
    crf_params_t* params = crfvot->params;
    params->addref(params);
    return params;
}


int crfvol_create_instance(const char *interface, void **ptr)
{
    if (strcmp(interface, "trainer.crfvo") == 0) {
        crf_trainer_t* trainer = (crf_trainer_t*)calloc(1, sizeof(crf_trainer_t));

        trainer->nref = 1;
        trainer->addref = crf_train_addref;
        trainer->release = crf_train_release;

        trainer->params = crf_train_params;
    
        trainer->set_message_callback = crf_train_set_message_callback;
        trainer->set_evaluate_callback = crf_train_set_evaluate_callback;
        trainer->train = crf_train_train;
        trainer->save = crf_train_save;
        trainer->internal = crfvol_new();

        *ptr = trainer;
        return 0;
    } else {
        return 1;
    }
}
