/*
 *      Linear-chain CRF tagger.
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

/* $Id: crfvo_tag.c 176 2010-07-14 09:31:04Z naoaki $ */

#ifdef    HAVE_CONFIG_H
#include <config.h>
#endif/*HAVE_CONFIG_H*/

#include <os.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <crfsuite.h>

#include "crfvo.h"

struct tag_crfvot {
    int num_labels;            /**< Number of distinct output labels (L). */
    int num_attributes;        /**< Number of distinct attributes (A). */
	int num_features;          /**< Number of features. */

    crfvom_t *model;        /**< CRF model. */
    crfvo_context_t *ctx;    /**< CRF context. */
};

crfvot_t *crfvot_new(crfvom_t* crfvom)
{
    crfvot_t* crfvot = NULL;

    crfvot = (crfvot_t*)calloc(1, sizeof(crfvot_t));
    crfvot->num_labels = crfvom_get_num_labels(crfvom);
    crfvot->num_attributes = crfvom_get_num_attrs(crfvom);
	crfvot->num_features = crfvom_get_num_features(crfvom);
    crfvot->model = crfvom;
    crfvot->ctx = crfvoc_new(crfvot->num_labels, 0, 0);

    return crfvot;
}

void crfvot_delete(crfvot_t* crfvot)
{
    crfvoc_delete(crfvot->ctx);
    free(crfvot);
}

int crfvot_tag(crfvot_t* crfvot, crf_sequence_t *inst, crf_output_t* output)
{
    int i;
    floatval_t score = 0;
    crfvo_context_t* ctx = crfvot->ctx;
	feature_refs_t* attrs;
	crfvol_feature_t* features;
	crfvopp_t* preprocessor;
	floatval_t* exp_weight;

	attrs = malloc(crfvot->num_attributes * sizeof(feature_refs_t));
	features = malloc(crfvot->num_features * sizeof(crfvol_feature_t));
	exp_weight = malloc(crfvot->num_features * sizeof(floatval_t));
	preprocessor = malloc(sizeof(crfvopp_t));
	if (!attrs || !features || !exp_weight || !preprocessor) return CRFERR_OUTOFMEMORY;

	crfvopp_new(preprocessor);

	for (i = 0; i < crfvot->num_attributes; ++i) {
		crfvom_get_attrref(crfvot->model, i, &attrs[i]);
	}

	for (i = 0; i < crfvot->num_features; ++i) {
		crfvom_feature_t f;
		crfvom_get_feature(crfvot->model, i, &f);
		features[i].attr = f.attr;
		memcpy(features[i].label_sequence, f.label_sequence, MAX_ORDER);
		features[i].order = f.order;
		exp_weight[i] = exp(f.weight);
	}

	crfvopp_preprocess_sequence(
		preprocessor,
		attrs,
		features,
		crfvot->num_labels,
		inst
		);

	crfvoc_set_num_items(ctx, inst->num_items, inst->max_paths);

	crfvoc_set_context(ctx, inst);
	crfvoc_set_weight(ctx, exp_weight);

    score = crfvoc_decode(ctx);

    crf_output_init_n(output, inst->num_items);
    output->probability = score;
    for (i = 0;i < inst->num_items;++i) {
        output->labels[i] = ctx->labels[i];
    }
    output->num_labels = inst->num_items;

	crfvopp_delete(preprocessor);
	free(preprocessor);
	free(attrs);
	free(features);
	free(exp_weight);

    return 0;
}
