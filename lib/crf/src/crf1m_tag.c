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

/* $Id: crf1m_tag.c 176 2010-07-14 09:31:04Z naoaki $ */

#ifdef    HAVE_CONFIG_H
#include <config.h>
#endif/*HAVE_CONFIG_H*/

#include <os.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <crfsuite.h>

#include "crf1m.h"

struct tag_crf1mt {
    int num_labels;            /**< Number of distinct output labels (L). */
    int num_attributes;        /**< Number of distinct attributes (A). */

    crf1mm_t *model;        /**< CRF model. */
    crf1m_context_t *ctx;    /**< CRF context. */
};

crf1mt_t *crf1mt_new(crf1mm_t* crf1mm)
{
    crf1mt_t* crf1mt = NULL;

    crf1mt = (crf1mt_t*)calloc(1, sizeof(crf1mt_t));
    crf1mt->num_labels = crf1mm_get_num_labels(crf1mm);
    crf1mt->num_attributes = crf1mm_get_num_attrs(crf1mm);
    crf1mt->model = crf1mm;
    crf1mt->ctx = crf1mc_new(crf1mt->num_labels, 0, 0);

    return crf1mt;
}

void crf1mt_delete(crf1mt_t* crf1mt)
{
    crf1mc_delete(crf1mt->ctx);
    free(crf1mt);
}

int crf1mt_tag(crf1mt_t* crf1mt, crf_sequence_t *inst, crf_output_t* output)
{
    int i;
    floatval_t score = 0;
    crf1m_context_t* ctx = crf1mt->ctx;

	for (i = 0; i < inst->num_items; ++i) {
		if (inst->items[i].preprocessed_data == 0) {
			crf1ml_preprocess_sequence(crf1mt, inst);
			break;
		}
	}
    crf1mc_set_num_items(ctx, inst->num_items, inst->max_paths);

    score = crf1mc_viterbi(ctx);

    crf_output_init_n(output, inst->num_items);
    output->probability = score;
    for (i = 0;i < inst->num_items;++i) {
        output->labels[i] = ctx->labels[i];
    }
    output->num_labels = inst->num_items;

    return 0;
}
