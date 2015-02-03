/*
 *      Feature generation for linear-chain CRF.
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

/* $Id: crfvo_feature.c 176 2010-07-14 09:31:04Z naoaki $ */

#ifdef    HAVE_CONFIG_H
#include <config.h>
#endif/*HAVE_CONFIG_H*/

#include <os.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <crfsuite.h>

#include "logging.h"
#include "crfvo.h"
#include "rumavl.h"    /* AVL tree library necessary for feature generation. */

/**
 * Feature set.
 */
struct tag_featureset {
    RUMAVL* avl;    /**< Root node of the AVL tree. */
    int num;        /**< Number of features in the AVL tree. */
};

#define    COMP(a, b)    ((a)>(b))-((a)<(b))

static int featureset_comp(const void *x, const void *y, size_t n, void *udata)
{
    int ret = 0;
    int i;
    const crfvol_feature_t* f1 = (const crfvol_feature_t*)x;
    const crfvol_feature_t* f2 = (const crfvol_feature_t*)y;

    ret = COMP(f1->attr, f2->attr);
    if (ret == 0) {
        ret = COMP(f1->order, f2->order);
        if (ret == 0) {
            for (i = 0; i < f1->order; ++i) {
                ret = COMP(f1->label_sequence[i], f2->label_sequence[i]);
                if (ret) break;
            }
        }
    }
    return ret;
}

featureset_t* featureset_new()
{
    featureset_t* set = NULL;
    set = (featureset_t*)calloc(1, sizeof(featureset_t));
    if (set != NULL) {
        set->num = 0;
        set->avl = rumavl_new(
            sizeof(crfvol_feature_t), featureset_comp, NULL, NULL);
        if (set->avl == NULL) {
            free(set);
            set = NULL;
        }
    }
    return set;
}

void featureset_delete(featureset_t* set)
{
    if (set != NULL) {
        rumavl_destroy(set->avl);
        free(set);
    }
}

static int featureset_add(featureset_t* set, const crfvol_feature_t* f)
{
    /* Check whether if the feature already exists. */
    crfvol_feature_t *p = (crfvol_feature_t*)rumavl_find(set->avl, f);
    if (p == NULL) {
        /* Insert the feature to the feature set. */
        rumavl_insert(set->avl, f);
        ++set->num;
    } else {
        /* An existing feature: add the observation expectation. */
        p->freq += f->freq;
    }
    return 0;
}

void featureset_generate(crfvol_features_t* features, featureset_t* set)
{
    int n = 0, k = 0;
    RUMAVL_NODE *node = NULL;
    crfvol_feature_t *f = NULL;

    features->features = 0;

    /* The first pass: count the number of valid features. */
    while ((node = rumavl_node_next(set->avl, node, 1, (void**)&f)) != NULL) {
        ++n;
    }

    /* The second path: copy the valid features to the feature array. */
    features->features = (crfvol_feature_t*)calloc(n, sizeof(crfvol_feature_t));
    if (features->features != NULL) {
        node = NULL;
        while ((node = rumavl_node_next(set->avl, node, 1, (void**)&f)) != NULL) {
            memcpy(&features->features[k], f, sizeof(crfvol_feature_t));
            ++k;
        }
        features->num_features = n;
    }
}

void crfvol_features_delete(crfvol_features_t* features)
{
    if (features) {
        free(features->features);
    }
    free(features);
}


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

int crfvol_add_feature(
    featureset_t* featureset,
    int attr,
    int order,
    unsigned char label_sequence[]
    )
{
    int i, ret;
    crfvol_feature_t f;

    memset(&f, 0, sizeof(f));
    f.attr = attr;
    f.order = order;
    for (i = 0; i < order; ++i) {
        f.label_sequence[i] = label_sequence[i];
    }
    ret = featureset_add(featureset, &f);
    if (ret < 0) return ret;
    return featureset->num;
}
