/*
 *      Forward backward algorithm of linear-chain CRF.
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

/* $Id: crf1m_context.c 176 2010-07-14 09:31:04Z naoaki $ */

#ifdef __cplusplus
extern "C" {
#endif

#ifdef    HAVE_CONFIG_H
#include <config.h>
#endif/*HAVE_CONFIG_H*/

#include <os.h>

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#include <crfsuite.h>

#include "crf1m.h"

crf1m_context_t* crf1mc_new(int L, int T, int max_paths)
{
    int ret = 0;
    crf1m_context_t* ctx = NULL;
    
    ctx = (crf1m_context_t*)calloc(1, sizeof(crf1m_context_t));
    if (ctx != NULL) {
        ctx->num_labels = L;
        ctx->norm_significand = 0.0;
		ctx->norm_exponent = 0;
		ctx->max_paths = -1;

        if (ret = crf1mc_set_num_items(ctx, T, max_paths)) {
            goto error_exit;
        }
        ctx->num_items = 0;
    }
    return ctx;

error_exit:
    crf1mc_delete(ctx);
    return NULL;
}

int crf1mc_set_num_items(crf1m_context_t* ctx, int T, int max_paths)
{
	int i;
    const int L = ctx->num_labels;

    ctx->num_items = T;
	if (ctx->max_paths < max_paths) {
		for (i = 0; i < ctx->max_items; ++i) {
			free(ctx->path_scores[i]);
			ctx->path_scores[i] = (crf1m_path_score_t*)calloc(max_paths, sizeof(crf1m_path_score_t));
	        if (ctx->path_scores[i] == NULL) return CRFERR_OUTOFMEMORY;
		}
		free(ctx->cur_temp_scores);
		free(ctx->prev_temp_scores);
		ctx->cur_temp_scores = (floatval_t*)calloc(max_paths, sizeof(floatval_t));
		ctx->prev_temp_scores = (floatval_t*)calloc(max_paths, sizeof(floatval_t));
		ctx->max_paths = max_paths;
	}

    if (ctx->max_items < T) {
		int i;
		crf1m_path_score_t** path_scores_new = (crf1m_path_score_t**)malloc((T+1) * sizeof(crf1m_path_score_t*));
		int** num_paths_by_label_new = (int**)malloc((T+1) * sizeof(int*));
        if (path_scores_new == NULL || num_paths_by_label_new == NULL) return CRFERR_OUTOFMEMORY;
		memcpy(path_scores_new, ctx->path_scores, sizeof(crf1m_path_score_t*) * (ctx->max_items));
		memcpy(num_paths_by_label_new, ctx->num_paths_by_label, sizeof(int*) * (ctx->max_items));
		for (i = ctx->max_items; i < T; ++i) {
			path_scores_new[i] = (crf1m_path_score_t*)calloc(max_paths, sizeof(crf1m_path_score_t));
			num_paths_by_label_new[i] = (int*)calloc(L, sizeof(int));
	        if (path_scores_new[i] == NULL || num_paths_by_label_new[i] == NULL) return CRFERR_OUTOFMEMORY;
		}
		
        free(ctx->exponents);
        free(ctx->labels);
		free(ctx->path_scores);
		free(ctx->fids_refs);
		free(ctx->num_paths);
		free(ctx->num_paths_by_label);
		free(ctx->training_path_indexes);
		ctx->path_scores = path_scores_new;
		ctx->num_paths_by_label = num_paths_by_label_new;

        ctx->labels = (int*)calloc(T, sizeof(int));
        ctx->exponents = (int*)calloc(T, sizeof(int));
		ctx->fids_refs = (int**)calloc(T, sizeof(int*));
		ctx->num_paths = (int*)calloc(T, sizeof(int));
		ctx->training_path_indexes = (int*)calloc(T, sizeof(int));
        if (ctx->labels == NULL || ctx->exponents == NULL ||
			ctx->fids_refs == NULL || ctx->num_paths == NULL) return CRFERR_OUTOFMEMORY;

        ctx->max_items = T;
    }

    return 0;
}

void crf1mc_delete(crf1m_context_t* ctx)
{
    if (ctx != NULL) {
		int i;
		for (i = 0; i < ctx->max_items; ++i) {
			free(ctx->path_scores[i]);
		}
		free(ctx->path_scores);
        free(ctx->exponents);
        free(ctx->labels);
		free(ctx->fids_refs);
		free(ctx->num_paths);
		free(ctx->cur_temp_scores);
		free(ctx->prev_temp_scores);
    }
    free(ctx);
}

floatval_t crf1mc_logprob(crf1m_context_t* ctx)
{
    int t;
    floatval_t ret = 0;
    const int T = ctx->num_items;

	for (t = 0; t < T; ++t) {
		ret += log(ctx->path_scores[t][ctx->training_path_indexes[t]].exp_weight);
	}
	ret -= log(ctx->norm_significand) + log(2.0) * ctx->norm_exponent;

    return ret;
}

floatval_t crf1mc_viterbi(crf1m_context_t* ctx)
{
	int T = ctx->num_items;
	int L = ctx->num_labels;
	floatval_t* prev_temp_scores = ctx->prev_temp_scores;
	floatval_t* cur_temp_scores = ctx->cur_temp_scores;
	floatval_t* prev_temp_scores_backup = (floatval_t*)malloc(sizeof(floatval_t) * ctx->max_paths);
	int* real_path_indexes = (int*)malloc(sizeof(int) * ctx->max_paths);

	int exponent_diff = 0;
	int exponent_all = 0;
	floatval_t real_scale_diff = 1.0;

	int n = 2, prev_n, i, j, t;

	floatval_t last_best_score = 0.0;
	int last_best_path = 0;

	prev_temp_scores[0] = 0.0;
	prev_temp_scores[1] = 1.0;

	for (t = 0; t < T; ++t) {
		int label, next_label, prev_index_start;
		floatval_t max_score;

		crf1m_path_score_t* path_scores = ctx->path_scores[t];
		prev_n = n;
		n = ctx->num_paths[t];
		memset(cur_temp_scores, 0, sizeof(floatval_t) * n);
		memcpy(prev_temp_scores_backup, prev_temp_scores, sizeof(floatval_t) * prev_n);
		for (i = 0; i < prev_n; ++i) real_path_indexes[i] = i;

		label = L;
		next_label = ctx->num_paths[t] - ctx->num_paths_by_label[t][L];
		prev_index_start = prev_n;
		max_score = 0.0;

		for (i = n-1; i > 0; --i) {
			int prev_path_index;

			while (i < next_label && label >= 0) {
				label--;
				next_label -= ctx->num_paths_by_label[t][label];
				memcpy(prev_temp_scores, prev_temp_scores_backup, sizeof(floatval_t) * prev_n);
				for (j = 0; j < prev_n; ++j) real_path_indexes[j] = j;
				prev_index_start = prev_n;
			}
			if (label < 0) break;
			prev_path_index = path_scores[i].path.prev_path_index;
			for (j = prev_index_start-1; j > prev_path_index; --j) {
				int longest_suffix_index = (t > 0) ? ctx->path_scores[t-1][j].path.longest_suffix_index : 0;
				if (prev_temp_scores[j] > prev_temp_scores[longest_suffix_index]) {
					prev_temp_scores[longest_suffix_index] = prev_temp_scores[j];
					real_path_indexes[longest_suffix_index] = real_path_indexes[j];
				}
			}
			prev_index_start = prev_path_index; 
			cur_temp_scores[i] = prev_temp_scores[prev_path_index] * path_scores[i].exp_weight;
			if (cur_temp_scores[i] > max_score) max_score = cur_temp_scores[i];
			ctx->path_scores[t][i].best_path = real_path_indexes[prev_path_index];
		}
		frexp(max_score, &exponent_diff);
		exponent_all += exponent_diff;
		real_scale_diff = ldexp(1.0, -exponent_diff);
		for (i = 1; i < n; ++i) {
			cur_temp_scores[i] *= real_scale_diff;
			path_scores[i].score = log(cur_temp_scores[i] * pow(2.0, exponent_all));
		}
		memcpy(prev_temp_scores, cur_temp_scores, sizeof(floatval_t) * n);
	}

	for (i = 1; i < n; ++i) {
		if (prev_temp_scores[i] > last_best_score) {
			last_best_score = prev_temp_scores[i];
			last_best_path = i;
		}
	}

	for (t = T-1; t >= 0; --t) {
		int label = 0;
		int path_num = 1 + ctx->num_paths_by_label[t][0];
		while (last_best_path >= path_num) {
			label++;
			path_num += ctx->num_paths_by_label[t][label];
		}
		ctx->labels[t] = label;
		last_best_path = ctx->path_scores[t][last_best_path].best_path;
	}
	free(prev_temp_scores_backup);
	free(real_path_indexes);
	return exponent_all * log(2.0) + log(last_best_score);
}

void crf1mc_debug_context(crf1m_context_t* ctx, FILE *fp)
{
    const floatval_t *fwd = NULL, *bwd = NULL;
    const floatval_t *state = NULL, *trans = NULL;
    const int T = ctx->num_items;
    const int L = ctx->num_labels;

    fprintf(fp, "# ===== Information =====\n");
    fprintf(fp, "NORM\t%fe%d\n", ctx->norm_significand, ctx->norm_exponent);
}

void crf1mc_test_context(FILE *fp)
{
	/*
    crf1m_context_t *ctx = crf1mc_new(3, 3);
    floatval_t *trans = NULL, *state = NULL;
    
    ctx->num_items = ctx->max_items;
    crf1mc_forward_score(ctx);
    crf1mc_backward_score(ctx);
    crf1mc_debug_context(ctx, fp);

    ctx->labels[0] = 0;    ctx->labels[1] = 2;    ctx->labels[2] = 0;
    printf("PROB\t%f\n", crf1mc_logprob(ctx));
	*/
}

void crf1mc_set_weight(
    crf1m_context_t* ctx,
    const floatval_t* exp_weight
	)
{
	int i, j, t;
	int T = ctx->num_items;

	for (t = 0; t < T; ++t) {
		crf1m_path_score_t* path_scores = ctx->path_scores[t];
		int* fids_ref = ctx->fids_refs[t];
		int n = ctx->num_paths[t];
		int fid_index = 0;

		// accumulate weight
		path_scores[0].exp_weight = 1.0;
		for (i = 1; i < n; ++i) {
			int feature_count = path_scores[i].path.feature_count;
			int longest_suffix_index = path_scores[i].path.longest_suffix_index;
			path_scores[i].exp_weight = 1.0;

			for (j = 0; j < feature_count; ++j) {
				path_scores[i].exp_weight *= exp_weight[fids_ref[fid_index++]];
			}
			path_scores[i].exp_weight *= path_scores[longest_suffix_index].exp_weight;
		}
	}
}

void crf1mc_accumulate_discount(
    crf1m_context_t* ctx
    )
{
	int i, last_n, t;
	int T = ctx->num_items;
	floatval_t* prev_temp_scores = ctx->prev_temp_scores;
	floatval_t* cur_temp_scores = ctx->cur_temp_scores;

	int exponent_diff = 0;
	floatval_t real_scale_diff = 1.0;

	prev_temp_scores[0] = 1.0; // gamma for empty path
	prev_temp_scores[1] = 1.0; // gamma for BOS

	ctx->exponents[0] = 0;

	// forward
	for (t = 0; t < T; ++t) {
		crf1m_path_score_t* path_scores = ctx->path_scores[t];
		int n = ctx->num_paths[t];

		memset(cur_temp_scores, 0, sizeof(floatval_t) * n);

		// forward scores
		real_scale_diff = ldexp(1.0, -exponent_diff);
		for (i = n-1; i > 0; --i) {
			int longest_suffix_index = path_scores[i].path.longest_suffix_index;
			int prev_path_index = path_scores[i].path.prev_path_index;
			floatval_t prev_gamma = prev_temp_scores[prev_path_index] * real_scale_diff;
			// alpha
			path_scores[longest_suffix_index].score -= prev_gamma;
			path_scores[i].score += prev_gamma;
			// gamma
			cur_temp_scores[i] += path_scores[i].score * path_scores[i].exp_weight;
			cur_temp_scores[longest_suffix_index] += cur_temp_scores[i];
		}
		path_scores[0].score = 0; // alpha for an empty path is 0
		frexp(cur_temp_scores[0], &exponent_diff);
		if (t < T-1) ctx->exponents[t+1] = ctx->exponents[t];
		ctx->exponents[t+1] += exponent_diff;
		memcpy(prev_temp_scores, cur_temp_scores, sizeof(floatval_t) * n);
	}
	real_scale_diff = ldexp(1.0, -exponent_diff);
	ctx->norm_significand = prev_temp_scores[0] * real_scale_diff;
	ctx->norm_exponent = ctx->exponents[T-1] + exponent_diff;	

	// backward / accumulate score
	last_n = ctx->num_paths[T-1];
	memset(cur_temp_scores, 0, sizeof(floatval_t) * last_n);
	cur_temp_scores[0] = real_scale_diff; // delta for the empty path
	for (t = T-1; t >= 0; --t) {
		crf1m_path_score_t* path_scores = ctx->path_scores[t];
		int n = ctx->num_paths[t];
		int prev_n = (t > 0) ? ctx->num_paths[t-1] : 2; // 2 paths (empty/BOS) for position 0
		memset(prev_temp_scores, 0, sizeof(floatval_t) * prev_n);

		// backward scores
		real_scale_diff = ldexp(1.0, (t > 0) ? (ctx->exponents[t-1] - ctx->exponents[t]) : 0);
		for (i = 1; i < n; ++i) {
			int longest_suffix_index = path_scores[i].path.longest_suffix_index;
			// beta
			cur_temp_scores[i] += cur_temp_scores[longest_suffix_index];
		}
		cur_temp_scores[0] = 0;
		for (i = 1; i < n; ++i) {
			int longest_suffix_index = path_scores[i].path.longest_suffix_index;
			int prev_path_index = path_scores[i].path.prev_path_index;
			// beta * W
			cur_temp_scores[i] *= path_scores[i].exp_weight;

			// theta (alpha * beta * W)
			path_scores[i].score *= cur_temp_scores[i];

			// delta
			prev_temp_scores[prev_path_index] +=
				(cur_temp_scores[i] - 
				 cur_temp_scores[longest_suffix_index]) * real_scale_diff;
		}
		path_scores[0].score = 0.0;
		for (i = n-1; i > 0; --i) {
			int longest_suffix_index = path_scores[i].path.longest_suffix_index;
			// sigma
			path_scores[longest_suffix_index].score += path_scores[i].score;
		}
		for (i = 1; i < n; ++i) {
			// normalize
			path_scores[i].score /= ctx->norm_significand;
		}
		memcpy(cur_temp_scores, prev_temp_scores, sizeof(floatval_t) * prev_n);
	}
}

#ifdef __cplusplus
} // extern "C"
#endif
