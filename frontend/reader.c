/*
 *        Data reader.
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

/* $Id: reader.c 176 2010-07-14 09:31:04Z naoaki $ */

#include <os.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <crfsuite.h>
#include "iwa.h"

static int progress(FILE *fpo, int prev, int current)
{
    while (prev < current) {
        ++prev;
        if (prev % 2 == 0) {
            if (prev % 10 == 0) {
                fprintf(fpo, "%d", prev / 10);
                fflush(fpo);
            } else {
                fprintf(fpo, ".", prev / 10);
                fflush(fpo);
            }
        }
    }
    return prev;
}

void read_data(FILE *fpi, FILE *fpo, crf_data_t* data, crf_dictionary_t* attrs, crf_dictionary_t* labels)
{
    int lid = -1;
    crf_sequence_t inst;
    crf_item_t item;
    crf_content_t cont;
    iwa_t* iwa = NULL;
    const iwa_token_t* token = NULL;
    long filesize = 0, begin = 0, offset = 0;
    int prev = 0, current = 0;

    /* Initialize the instance.*/
    crf_sequence_init(&inst);

    /* Obtain the file size. */
    begin = ftell(fpi);
    fseek(fpi, 0, SEEK_END);
    filesize = ftell(fpi) - begin;
    fseek(fpi, begin, SEEK_SET);

    /* */
    fprintf(fpo, "0");
    fflush(fpo);
    prev = 0;

    iwa = iwa_reader(fpi);
    while (token = iwa_read(iwa), token != NULL) {
        /* Progress report. */
        offset = ftell(fpi);
        current = (int)((offset - begin) * 100.0 / (double)filesize);
        prev = progress(fpo, prev, current);

        switch (token->type) {
        case IWA_BOI:
            /* Initialize an item. */
            lid = -1;
            crf_item_init(&item);
            break;
        case IWA_EOI:
            /* Append the item to the instance. */
            crf_sequence_append(&inst, &item, lid);
            crf_item_finish(&item);
            break;
        case IWA_ITEM:
            if (lid == -1) {
                if (!strcmp(token->attr, "__BOS_EOS__")) {
                    lid = -2; /* EOS : to be overwritten by L */
                } else {
                    lid = labels->get(labels, token->attr);
                }
            } else {
                crf_content_init(&cont);
                cont.aid = attrs->get(attrs, token->attr);
                if (token->value && *token->value) {
                    cont.scale = atof(token->value);
                } else {
                    cont.scale = 1.0;
                }
                crf_item_append_content(&item, &cont);
            }
            break;
        case IWA_NONE:
        case IWA_EOF:
            /* Put the training instance. */
            crf_data_append(data, &inst);
            crf_sequence_finish(&inst);
            break;
        case IWA_COMMENT:
            break;
        }
    }
    progress(fpo, prev, 100);
    fprintf(fpo, "\n");
    iwa_delete(iwa);
}

int read_features(
    FILE* fpi,
    FILE* fpo,
    crf_dictionary_t* labels,
    crf_dictionary_t* attrs,
    crf_trainer_t* trainer
    )
{
    const iwa_token_t* token = NULL;
    iwa_t* iwa = NULL;
    long filesize = 0, begin = 0, offset = 0;
    int prev = 0, current = 0;
    int L = labels->num(labels);
    int attr = -1;
    int order = 0;
    unsigned char* label_sequence;
    int max_order = 256;
    int ret = 0;

    label_sequence = calloc(max_order, sizeof(unsigned char));

    /* Obtain the file size. */
    begin = ftell(fpi);
    fseek(fpi, 0, SEEK_END);
    filesize = ftell(fpi) - begin;
    fseek(fpi, begin, SEEK_SET);

    fprintf(fpo, "0");
    fflush(fpo);
    prev = 0;

    /* Loop over the sequences in the training data. */

    iwa = iwa_reader(fpi);
    while (token = iwa_read(iwa), token != NULL) {
        /* Progress report. */
        int offset = ftell(fpi);
        current = (int)((offset - begin) * 100.0 / (double)filesize);
        prev = progress(fpo, prev, current);

        switch (token->type) {
        case IWA_BOI:
            /* Initialize a feature. */
            memset(label_sequence, 0, max_order * sizeof(unsigned char));
            attr = -1;
            order = 0;
            break;
        case IWA_EOI:
            /* Append the feature to the feature set. */
            if (attr != -1 && order != -1) ret = trainer->add_feature(trainer, attr, order, label_sequence);
            break;
        case IWA_ITEM:
            if (attr == -1) {
                attr = attrs->get(attrs, token->attr);
            } else {
                int label = labels->to_id(labels, token->attr);
                if (label < 0) label = L;
                if (order >= max_order) {
                    unsigned char* t = realloc(label_sequence, max_order * 2);
                    if (t == 0) {
                        free(label_sequence);
                        return CRFERR_OUTOFMEMORY;
                    }
                    label_sequence = t;
                }
                label_sequence[order] = label;
                order++;
            }
            break;
        case IWA_NONE:
        case IWA_EOF:
            break;
        case IWA_COMMENT:
            break;
        }
    }
    progress(fpo, prev, 100);
    fprintf(fpo, "\n");
    iwa_delete(iwa);

    free(label_sequence);
    return ret;
}
