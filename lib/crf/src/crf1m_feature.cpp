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

/* $Id: crf1m_feature.c 176 2010-07-14 09:31:04Z naoaki $ */

#include "esa.hxx"
#include <vector>
using namespace std;

#ifdef __cplusplus
extern "C" {
#endif

#ifdef    HAVE_CONFIG_H
#include <config.h>
#endif/*HAVE_CONFIG_H*/

#include <os.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <crfsuite.h>

#include "logging.h"
#include "crf1m.h"
#include "rumavl.h"    /* AVL tree library necessary for feature generation. */

/**
 * Feature set.
 */
typedef struct {
    RUMAVL* avl;    /**< Root node of the AVL tree. */
    int num;        /**< Number of features in the AVL tree. */
} featureset_t;


#define    COMP(a, b)    ((a)>(b))-((a)<(b))

static int featureset_comp(const void *x, const void *y, size_t n, void *udata)
{
    int ret = 0;
	int i;
    const crf1ml_feature_t* f1 = (const crf1ml_feature_t*)x;
    const crf1ml_feature_t* f2 = (const crf1ml_feature_t*)y;

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

static featureset_t* featureset_new()
{
    featureset_t* set = NULL;
    set = (featureset_t*)calloc(1, sizeof(featureset_t));
    if (set != NULL) {
        set->num = 0;
        set->avl = rumavl_new(
            sizeof(crf1ml_feature_t), featureset_comp, NULL, NULL);
        if (set->avl == NULL) {
            free(set);
            set = NULL;
        }
    }
    return set;
}

static void featureset_delete(featureset_t* set)
{
    if (set != NULL) {
        rumavl_destroy(set->avl);
        free(set);
    }
}

static int featureset_add(featureset_t* set, const crf1ml_feature_t* f)
{
    /* Check whether if the feature already exists. */
    crf1ml_feature_t *p = (crf1ml_feature_t*)rumavl_find(set->avl, f);
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

static void featureset_generate(crf1ml_features_t* features, featureset_t* set, floatval_t minfreq)
{
    int n = 0, k = 0;
    RUMAVL_NODE *node = NULL;
    crf1ml_feature_t *f = NULL;

    features->features = 0;

    /* The first pass: count the number of valid features. */
    while ((node = rumavl_node_next(set->avl, node, 1, (void**)&f)) != NULL) {
        if (minfreq <= f->freq) {
            ++n;
        }
    }

    /* The second path: copy the valid features to the feature array. */
    features->features = (crf1ml_feature_t*)calloc(n, sizeof(crf1ml_feature_t));
    if (features->features != NULL) {
        node = NULL;
        while ((node = rumavl_node_next(set->avl, node, 1, (void**)&f)) != NULL) {
            if (minfreq <= f->freq) {
                memcpy(&features->features[k], f, sizeof(crf1ml_feature_t));
                ++k;
            }
        }
        features->num_features = n;
    }
}

crf1ml_features_t* crf1ml_generate_features(
    const crf_sequence_t *seqs,
    int num_sequences,
    int num_labels,
    int num_attributes,
	int emulate_crf1m,
    floatval_t minfreq,
    crf_logging_callback func,
    void *instance
    )
{
    int i, j, s, t, c;
    crf1ml_feature_t f;
    featureset_t* set = NULL;
    crf1ml_features_t *features = NULL;
    const int N = num_sequences;
    const int L = num_labels;
    logging_t lg;

    lg.func = func;
    lg.instance = instance;
    lg.percent = 0;

    /* Allocate a feature container. */
    features = (crf1ml_features_t*)calloc(1, sizeof(crf1ml_features_t));

    /* Create an instance of feature set. */
    set = featureset_new();

    /* Loop over the sequences in the training data. */
    logging_progress_start(&lg);

	if (emulate_crf1m) {
		for (s = 0; s < N; ++s) {
			int prev = L, cur = 0;
			const crf_item_t* item = NULL;
			const crf_sequence_t* seq = &seqs[s];
			const int T = seq->num_items;
			uint8_t* label_sequence = (uint8_t*)malloc((T+1) * sizeof(uint8_t));
			label_sequence[T] = L; // BOS

			for (t = 0; t < T; ++t) {
				int a;
				int label_sequence_pos = T - t - 1;
				item = &seq->items[t];
				label_sequence[label_sequence_pos] = item->label;

				memset(&f, 0, sizeof(f));
				memcpy(f.label_sequence, &label_sequence[label_sequence_pos], 2);
				f.order = 2;
				f.freq = 1;
				f.attr = item->contents[0].aid;
				featureset_add(set, &f);

				f.attr = item->contents[1].aid;
				featureset_add(set, &f);

				f.attr = item->contents[2].aid;
				featureset_add(set, &f);

				if (t >= 1) {
					f.label_sequence[2] = label_sequence[label_sequence_pos+2];
					f.order = 3;
					featureset_add(set, &f);
					
					f.attr = item->contents[4].aid;
					featureset_add(set, &f);

					f.attr = item->contents[6].aid;
					featureset_add(set, &f);

					if (t >= 2) {
						f.label_sequence[3] = label_sequence[label_sequence_pos+3];
						f.order = 4;
						featureset_add(set, &f);
					}
				}

				if (t < T-1) { // t == T-1 : EOS
					for (a = 0; a < item->num_contents; ++a) {
						memset(&f, 0, sizeof(f));
						f.attr = item->contents[a].aid;
						f.order = 1;
						f.label_sequence[0] = label_sequence[label_sequence_pos];
						f.freq = 1;
						featureset_add(set, &f);
					}
				}
			}
			free(label_sequence);
		}
		for (i = 0; i < L; ++i) {
			// BOS
			memset(&f, 0, sizeof(f));
			f.label_sequence[0] = i;
			f.label_sequence[1] = L;
			f.order = 2;
			f.freq = 0;
			f.attr = 0;
			featureset_add(set, &f);

			// EOS
			memset(&f, 0, sizeof(f));
			f.label_sequence[0] = L;
			f.label_sequence[1] = i;
			f.order = 2;
			f.freq = 0;
			f.attr = 0;
			featureset_add(set, &f);
		}
	} else {
		vector<int> label_vector;
		vector<const crf_item_t*> item_pointer_vector;

		for (s = 0; s < N; ++s) {
			int prev = L, cur = 0;
			const crf_item_t* item = NULL;
			const crf_sequence_t* seq = &seqs[s];
			const int T = seq->num_items;

			/* BOS */
			label_vector.push_back(L);
			item_pointer_vector.push_back(0);

			/* Loop over the items in the sequence. */
			for (t = 0;t < T;++t) {
				item = &seq->items[t];
				label_vector.push_back(item->label); // t == T-1 : EOS
				item_pointer_vector.push_back(item);
			}

			logging_progress(&lg, s * 100 / N);
		}
		logging_progress_end(&lg);

		int size = label_vector.size();
		vector<int> SA(size);
		vector<int> left (size);
		vector<int> right (size);
		vector<int> D (size);
		int k = L + 1;
		int nodeNum = 0;
		int ret = esaxx(label_vector.begin(), SA.begin(), 
			left.begin(), right.begin(), D.begin(), 
			(int)label_vector.size(), k, nodeNum);

		vector<int> rank (size);
		int r = 0;
		for (i = 0; i < size; i++) {
			if (i == 0 || SA[i] == 0 || SA[i-1] == 0 || label_vector[(SA[i] + size - 1) % size] != label_vector[(SA[i - 1] + size - 1) % size]) r++;
			rank[i] = r;
		}

		int* attribute_count_array = new int[num_attributes];
		int* appeared_attributes_array = new int[num_attributes];
		memset(attribute_count_array, 0, num_attributes * sizeof(int));

		for (i = 0; i < nodeNum; i++) {
			if (rank[right[i] - 1] - rank[left[i]] > 0 && D[i] <= MAX_ORDER) {
				int length = D[i];
				if (length == 0) continue;
				char label_sequence[MAX_ORDER] = {0};
				int l = SA[left[i]];
				int r = SA[left[i]] + length - 1;

				int current_position, last_position;
				current_position = last_position = r;
				bool is_first = true;

				while (last_position >= l) {
					for (current_position = last_position;
						current_position > l &&
						!(current_position > 0 &&
						  label_vector[current_position] == L &&
						  label_vector[current_position-1] == L
						);
						--current_position);
					is_first = false;
					current_position--;
					int order = last_position - current_position;

					for (j = 0; j < order; ++j) {
						label_sequence[j] = label_vector[last_position - j]; // copy label sequence(reversed)
					}
					int   appeared_attributes_count = 0;
					for (j = left[i]; j < right[i]; ++j) {
						const crf_item_t* item = item_pointer_vector[SA[j] + last_position - l];
						if (item == NULL) continue; // BOS
						for (c = 0; c < item->num_contents; c++) {
							int aid = item->contents[c].aid;
							if (attribute_count_array[aid] == 0) {
								appeared_attributes_array[appeared_attributes_count] = aid;
								appeared_attributes_count++;
							}
							attribute_count_array[aid]++;
						}
					}
					for (j = 0; j < appeared_attributes_count; ++j) {
						if (attribute_count_array[appeared_attributes_array[j]] > 1) {
							f.order = order;
							f.freq = attribute_count_array[appeared_attributes_array[j]];
							f.attr = appeared_attributes_array[j];
							memcpy(f.label_sequence, label_sequence, MAX_ORDER * sizeof(char));
							featureset_add(set, &f);
						}
						attribute_count_array[appeared_attributes_array[j]] = 0;
					}
					last_position = current_position;
				}
			}
		}
		delete[] attribute_count_array;
		delete[] appeared_attributes_array;
	}

    /* Convert the feature set to an feature array. */
    featureset_generate(features, set, minfreq);

    /* Delete the feature set. */
    featureset_delete(set);

    return features;
}

#ifdef __cplusplus
} // extern "C"
#endif
