/*
 *      Linear-chain Conditional Random Fields (CRF).
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

/* $Id: crf1m.h 168 2010-01-29 05:46:24Z naoaki $ */

#ifndef    __CRF1M_H__
#define    __CRF1M_H__

#include <time.h>
#include <stdint.h>
#include "logging.h"

#define MAX_ORDER 8

typedef struct {
	crf_path_t path;
	floatval_t score;
	floatval_t exp_weight;
	int    best_path;
} crf1m_path_score_t;

// preprocessed data
typedef struct {
	int                num_paths;
	crf_path_t*        paths;
	int*               num_paths_by_label;
	int                training_path_index;
	int                num_fids;
	int*               fids;
} crf1m_preprocessed_data_t;

/**
 * CRF context. 
 */

typedef struct {
    /**
     * The total number of distinct output labels.
     *    The label number #num_labels represents BOS/EOS.
     */
    int num_labels;

    /**
     * The number of items.
     */
    int num_items;

    /**
     * The maximum number of labels.
     */
    int max_items;

    /**
     * Label array.
     *    This is a [T] vector whose element [t] presents the output label
     *    at position #t.
     */
    int *labels;

	int max_paths;
	crf1m_path_score_t** path_scores; // alpha -> alpha * beta -> sigma
	int*  num_paths;
	int*  training_path_indexes;
	int** num_paths_by_label;
	int** fids_refs;
	floatval_t* cur_temp_scores;  // beta * W (backward)
	floatval_t* prev_temp_scores; // gamma (forward) / delta (backward)
    /**
     * The normalize factor for the input sequence.
     *    This is equivalent to the total scores of all paths from BOS to
     *    EOS, given an input sequence.
     */
    floatval_t norm_significand;
	int norm_exponent;

	// exponents of scores
    int *exponents;

} crf1m_context_t;

/* crf1m_common.c */
crf1m_context_t* crf1mc_new(int L, int T, int max_paths);
int crf1mc_set_num_items(crf1m_context_t* ctx, int T, int max_paths);
void crf1mc_delete(crf1m_context_t* ctx);
void crf1mc_set_weight(crf1m_context_t* ctx, const floatval_t* exp_weight);
void crf1mc_accumulate_discount(crf1m_context_t* ctx);
floatval_t crf1mc_logprob(crf1m_context_t* ctx);
floatval_t crf1mc_viterbi(crf1m_context_t* ctx);
void crf1mc_debug_context(crf1m_context_t* ctx, FILE *fp);
void crf1mc_test_context(FILE *fp);


/**
 * A feature (for either state or transition).
 */
typedef struct {
    /**
     *    Order of the feature function.
     */
    int        order;

    /**
     *    Attribute id.
     */
    int        attr;

    /**
     *    
     *    Label sequence.
     */
    uint8_t    label_sequence[MAX_ORDER];

    /**
     * Frequency (observation expectation).
     */
    floatval_t    freq;
} crf1ml_feature_t;

/**
 * Feature set.
 */
typedef struct {
    int                    num_features;    /**< Number of features. */
    crf1ml_feature_t*    features;        /**< Array of features. */
} crf1ml_features_t;

/**
 * Feature references.
 *    This is a collection of feature ids used for faster accesses.
 */
typedef struct {
    int        num_features;    /**< Number of features referred */
    int*    fids;            /**< Array of feature ids */
} feature_refs_t;

crf1ml_features_t* crf1ml_generate_features(
    const crf_sequence_t *seqs,
    int num_sequences,
    int num_labels,
    int num_attributes,
	int emulate_crf1m,
    floatval_t minfreq,
    crf_logging_callback func,
    void *instance
    );

/* crf1m_model.c */
struct tag_crf1mm;
typedef struct tag_crf1mm crf1mm_t;

struct tag_crf1mmw;
typedef struct tag_crf1mmw crf1mmw_t;

typedef struct {
    int        order;
    int        attr;
    uint8_t    label_sequence[MAX_ORDER];
    floatval_t    weight;
} crf1mm_feature_t;

crf1mmw_t* crf1mmw(const char *filename);
int crf1mmw_close(crf1mmw_t* writer);
int crf1mmw_open_labels(crf1mmw_t* writer, int num_labels);
int crf1mmw_close_labels(crf1mmw_t* writer);
int crf1mmw_put_label(crf1mmw_t* writer, int lid, const char *value);
int crf1mmw_open_attrs(crf1mmw_t* writer, int num_attributes);
int crf1mmw_close_attrs(crf1mmw_t* writer);
int crf1mmw_put_attr(crf1mmw_t* writer, int aid, const char *value);
int crf1mmw_open_labelrefs(crf1mmw_t* writer, int num_labels);
int crf1mmw_close_labelrefs(crf1mmw_t* writer);
int crf1mmw_put_labelref(crf1mmw_t* writer, int lid, const feature_refs_t* ref, int *map);
int crf1mmw_open_attrrefs(crf1mmw_t* writer, int num_attrs);
int crf1mmw_close_attrrefs(crf1mmw_t* writer);
int crf1mmw_put_attrref(crf1mmw_t* writer, int aid, const feature_refs_t* ref, int *map);
int crf1mmw_open_features(crf1mmw_t* writer);
int crf1mmw_close_features(crf1mmw_t* writer);
int crf1mmw_put_feature(crf1mmw_t* writer, int fid, const crf1mm_feature_t* f);


crf1mm_t* crf1mm_new(const char *filename);
void crf1mm_close(crf1mm_t* model);
int crf1mm_get_num_attrs(crf1mm_t* model);
int crf1mm_get_num_labels(crf1mm_t* model);
const char *crf1mm_to_label(crf1mm_t* model, int lid);
int crf1mm_to_lid(crf1mm_t* model, const char *value);
int crf1mm_to_aid(crf1mm_t* model, const char *value);
const char *crf1mm_to_attr(crf1mm_t* model, int aid);
int crf1mm_get_labelref(crf1mm_t* model, int lid, feature_refs_t* ref);
int crf1mm_get_attrref(crf1mm_t* model, int aid, feature_refs_t* ref);
int crf1mm_get_featureid(feature_refs_t* ref, int i);
int crf1mm_get_feature(crf1mm_t* model, int fid, crf1mm_feature_t* f);
void crf1mm_dump(crf1mm_t* model, FILE *fp);


typedef struct {
    char*        regularization;
    floatval_t    regularization_sigma;
    int            memory;
    floatval_t    epsilon;
    int         stop;
    floatval_t  delta;
    int            max_iterations;
    char*       linesearch;
    int         linesearch_max_iterations;
} crf1ml_lbfgs_option_t;

typedef struct {
    char*       algorithm;
    floatval_t    feature_minfreq;
	int           feature_emulate_crf1m;

    crf1ml_lbfgs_option_t   lbfgs;
} crf1ml_option_t;


/**
 * First-order Markov CRF trainer.
 */
struct tag_crf1ml {
    int num_labels;            /**< Number of distinct output labels (L). */
    int num_attributes;        /**< Number of distinct attributes (A). */

    int max_items;
	int max_paths;

    int num_sequences;
    crf_sequence_t* seqs;
    crf_tagger_t tagger;

    crf1m_context_t *ctx;    /**< CRF context. */

    logging_t* lg;

    void *cbe_instance;
    crf_evaluate_callback cbe_proc;

    feature_refs_t* attributes;

    int num_features;            /**< Number of distinct features (K). */

    /**
     * Feature array.
     *    Elements must be sorted by type, src, and dst in this order.
     */
    crf1ml_feature_t *features;

    floatval_t *w;            /**< Array of w (feature weights) */
	floatval_t *exp_weight;
    floatval_t *prob;

    crf_params_t* params;
    crf1ml_option_t opt;

    clock_t clk_begin;
    clock_t clk_prev;

    void *solver_data;

	void *preprocessor_data;
	void (*preprocessor_data_delete_func)(void*);
};
typedef struct tag_crf1ml crf1ml_t;

typedef void (*update_feature_t)(
    crf1ml_feature_t* f,
    const int fid,
    floatval_t prob,
    floatval_t scale,
    crf1ml_t* trainer,
    const crf_sequence_t* seq,
    int t
    );

void crf1ml_preprocess(crf1ml_t* trainer);
void crf1ml_enum_features(crf1ml_t* trainer, const crf_sequence_t* seq, update_feature_t func, double* logp);
void crf1ml_shuffle(int *perm, int N, int init);
void crf1ml_set_context(crf1ml_t* trainer, const crf_sequence_t* seq);
void crf1ml_preprocess_sequence(crf1ml_t* trainer, crf_sequence_t* seq);

/* crf1m_learn_lbfgs.c */
int crf1ml_lbfgs(crf1ml_t* crf1mt, crf1ml_option_t *opt);
int crf1ml_lbfgs_options(crf_params_t* params, crf1ml_option_t* opt, int mode);

/* crf1m_tag.c */
struct tag_crf1mt;
typedef struct tag_crf1mt crf1mt_t;

crf1mt_t *crf1mt_new(crf1mm_t* crf1mm);
void crf1mt_delete(crf1mt_t* crf1mt);
int crf1mt_tag(crf1mt_t* crf1mt, crf_sequence_t *inst, crf_output_t* output);


#endif/*__CRF1M_H__*/
