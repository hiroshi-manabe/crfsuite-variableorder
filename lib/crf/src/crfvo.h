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

/* $Id: crfvo.h 168 2010-01-29 05:46:24Z naoaki $ */

#ifndef    __CRFVO_H__
#define    __CRFVO_H__

#include <time.h>
#include <stdint.h>
#include "logging.h"

#define MAX_ORDER 8

typedef struct {
	int    prev_path_index;
	int    longest_suffix_index;
	int    feature_count;
} crfvo_path_t;

typedef struct {
	crfvo_path_t path;
	floatval_t score;
	floatval_t exp_weight;
	int    best_path;
} crfvo_path_score_t;

/*
	Preprocessed data.
*/
typedef struct {
	int                num_paths;
	crfvo_path_t*      paths;
	int*               num_paths_by_label;
	int                training_path_index;
	int                num_fids;
	int*               fids;
} crfvopd_t;

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
	crfvo_path_score_t** path_scores; /* alpha -> alpha * beta -> sigma */
	int*  num_paths;
	int*  training_path_indexes;
	int** num_paths_by_label;
	int** fids_refs;
	floatval_t* cur_temp_scores;  /* beta * W (backward) */
	floatval_t* prev_temp_scores; /* gamma (forward) / delta (backward) */
    /**
     * The normalize factor for the input sequence.
     *    This is equivalent to the total scores of all paths from BOS to
     *    EOS, given an input sequence.
	 *    norm_significand * 2^(norm_exponent)
     */
    floatval_t norm_significand;
	int norm_exponent;

	/* exponents of scores */
    int *exponents;

} crfvo_context_t;

/* crfvo_common.c */
crfvo_context_t* crfvoc_new(int L, int T, int max_paths);
int crfvoc_set_num_items(crfvo_context_t* ctx, int T, int max_paths);
void crfvoc_set_context(crfvo_context_t* ctx, const crf_sequence_t* seq);
void crfvoc_delete(crfvo_context_t* ctx);
void crfvoc_set_weight(crfvo_context_t* ctx, const floatval_t* exp_weight);
void crfvoc_calc_feature_expectations(crfvo_context_t* ctx);
floatval_t crfvoc_logprob(crfvo_context_t* ctx);
floatval_t crfvoc_decode(crfvo_context_t* ctx);
void crfvoc_debug_context(crfvo_context_t* ctx, FILE *fp);
void crfvoc_test_context(FILE *fp);

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
} crfvol_feature_t;

/**
 * Feature set.
 */
typedef struct {
    int                    num_features;    /**< Number of features. */
    crfvol_feature_t*    features;        /**< Array of features. */
} crfvol_features_t;

/**
 * Feature references.
 *    This is a collection of feature ids used for faster accesses.
 */
typedef struct {
    int        num_features;    /**< Number of features referred */
    int*    fids;            /**< Array of feature ids */
} feature_refs_t;

crfvol_features_t* crfvol_read_features(
	FILE* fpi,
	crf_dictionary_t* labels,
    crf_dictionary_t* attrs,
    crf_logging_callback func,
    void *instance
    );

/* crfvo_model.c */
struct tag_crfvom;
typedef struct tag_crfvom crfvom_t;

struct tag_crfvomw;
typedef struct tag_crfvomw crfvomw_t;

typedef struct {
    int        order;
    int        attr;
    uint8_t    label_sequence[MAX_ORDER];
    floatval_t    weight;
} crfvom_feature_t;

crfvomw_t* crfvomw(const char *filename);
int crfvomw_close(crfvomw_t* writer);
int crfvomw_open_labels(crfvomw_t* writer, int num_labels);
int crfvomw_close_labels(crfvomw_t* writer);
int crfvomw_put_label(crfvomw_t* writer, int lid, const char *value);
int crfvomw_open_attrs(crfvomw_t* writer, int num_attributes);
int crfvomw_close_attrs(crfvomw_t* writer);
int crfvomw_put_attr(crfvomw_t* writer, int aid, const char *value);
int crfvomw_open_attrrefs(crfvomw_t* writer, int num_attrs);
int crfvomw_close_attrrefs(crfvomw_t* writer);
int crfvomw_put_attrref(crfvomw_t* writer, int aid, const feature_refs_t* ref, int *map);
int crfvomw_open_features(crfvomw_t* writer);
int crfvomw_close_features(crfvomw_t* writer);
int crfvomw_put_feature(crfvomw_t* writer, int fid, const crfvom_feature_t* f);


crfvom_t* crfvom_new(const char *filename);
void crfvom_close(crfvom_t* model);
int crfvom_get_num_attrs(crfvom_t* model);
int crfvom_get_num_labels(crfvom_t* model);
int crfvom_get_num_features(crfvom_t* model);
const char *crfvom_to_label(crfvom_t* model, int lid);
int crfvom_to_lid(crfvom_t* model, const char *value);
int crfvom_to_aid(crfvom_t* model, const char *value);
const char *crfvom_to_attr(crfvom_t* model, int aid);
int crfvom_get_attrref(crfvom_t* model, int aid, feature_refs_t* ref);
int crfvom_get_featureid(feature_refs_t* ref, int i);
int crfvom_get_feature(crfvom_t* model, int fid, crfvom_feature_t* f);
void crfvom_dump(crfvom_t* model, FILE *fp);


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
} crfvol_lbfgs_option_t;

typedef struct {
    char*       algorithm;

    crfvol_lbfgs_option_t   lbfgs;
} crfvol_option_t;


/* crfvo_preprocess.c */
crfvopd_t* crfvopd_new(int L, int num_paths, int num_fids);
void crfvopd_delete(crfvopd_t* pp);

struct tag_buffer_manager;
typedef struct tag_buffer_manager buffer_manager_t;

typedef struct tag_crfvopp {
	buffer_manager_t* path_manager;
	buffer_manager_t* node_manager;
	buffer_manager_t* fid_list_manager;
} crfvopp_t;

crfvopp_t* crfvopp_new();
void crfvopp_delete(crfvopp_t* pp);
void crfvopp_preprocess_sequence(
	crfvopp_t* pp,
	const feature_refs_t* attrs,
	const crfvol_feature_t* features,
	const int L,
	crf_sequence_t* seq);


/**
 * Variable-order Markov CRF trainer.
 */
struct tag_crfvol {
    int num_labels;            /**< Number of distinct output labels (L). */
    int num_attributes;        /**< Number of distinct attributes (A). */

    int max_items;
	int max_paths;

    int num_sequences;
    crf_sequence_t* seqs;
    crf_tagger_t tagger;

    crfvo_context_t *ctx;    /**< CRF context. */

    logging_t* lg;

    void *cbe_instance;
    crf_evaluate_callback cbe_proc;

    feature_refs_t* attributes;

    int num_features;            /**< Number of distinct features (K). */

    /**
     * Feature array.
     *    Elements must be sorted by type, src, and dst in this order.
     */
    crfvol_feature_t *features;

    floatval_t *w;            /**< Array of w (feature weights) */
	floatval_t *exp_weight;
    floatval_t *prob;

    crf_params_t* params;
    crfvol_option_t opt;

    clock_t clk_begin;
    clock_t clk_prev;

    void *solver_data;

	crfvopp_t *preprocessor;
};
typedef struct tag_crfvol crfvol_t;

typedef void (*update_feature_t)(
    crfvol_feature_t* f,
    const int fid,
    floatval_t prob,
    floatval_t scale,
    crfvol_t* trainer,
    const crf_sequence_t* seq,
    int t
    );

void crfvol_preprocess(crfvol_t* trainer);
void crfvol_enum_features(crfvol_t* trainer, const crf_sequence_t* seq, update_feature_t func, double* logp);
void crfvol_shuffle(int *perm, int N, int init);

/* crfvo_learn_lbfgs.c */
int crfvol_lbfgs(crfvol_t* crfvot, crfvol_option_t *opt);
int crfvol_lbfgs_options(crf_params_t* params, crfvol_option_t* opt, int mode);

/* crfvo_tag.c */
struct tag_crfvot;
typedef struct tag_crfvot crfvot_t;

crfvot_t *crfvot_new(crfvom_t* crfvom);
void crfvot_delete(crfvot_t* crfvot);
int crfvot_tag(crfvot_t* crfvot, crf_sequence_t *inst, crf_output_t* output);

#endif/*__CRFVO_H__*/
