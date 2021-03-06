/*
 *      Variable Order Linear-chain CRF preprocess.
 *
 * Copyright (c) 2011, Hiroshi Manabe
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

#include <os.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <crfsuite.h>
#include "crfvo.h"

crfvopd_t* crfvopd_new(int L, int num_paths, int num_fids)
{
    crfvopd_t* pd = (crfvopd_t*)calloc(1, sizeof(crfvopd_t));    

    pd->num_paths = num_paths;
    pd->num_fids = num_fids;
    pd->fids = (int*)malloc(sizeof(int) * num_fids);
    pd->paths = (crfvo_path_t*)malloc(sizeof(crfvo_path_t) * num_paths);
    pd->num_paths_by_label = (int*)malloc(sizeof(int) * (L+1));
    return pd;
}

void crfvopd_delete(crfvopd_t* pd)
{
    if (pd != NULL) {
        free(pd->fids);
        free(pd->paths);
        free(pd->num_paths_by_label);
    }
    free(pd);
}

struct tag_buffer_manager
{
    void* buffer;
    int unit_size;
    int buffer_max;
    int buffer_used;
};

void* buf_init(buffer_manager_t* manager, int unit_size, int initial_buffer_max)
{
    manager->unit_size = unit_size;
    manager->buffer_max = initial_buffer_max;
    manager->buffer_used = 0;
    manager->buffer = calloc(unit_size, initial_buffer_max);
    return manager->buffer;
}

void* buf_from_index(buffer_manager_t* manager, int index)
{
    return (char*)manager->buffer + manager->unit_size * index;
}

int buf_get_new_index(buffer_manager_t* manager, int count)
{
    int ret;
    while (manager->buffer_max < manager->buffer_used + count) {
        void* temp = realloc(manager->buffer, manager->buffer_max * 2 * manager->unit_size);
        if (!temp) return -1;
        manager->buffer = temp;
        manager->buffer_max *= 2;
        memset((char*)manager->buffer + manager->unit_size * manager->buffer_used, 0,
            manager->unit_size * (manager->buffer_max - manager->buffer_used));
    }
    ret = manager->buffer_used;
    manager->buffer_used += count;
    return ret;
}

int buf_get_current_index(buffer_manager_t* manager)
{
    return manager->buffer_used;
}

void buf_clear(buffer_manager_t* manager)
{
    memset(manager->buffer, 0, manager->unit_size * manager->buffer_used);
    manager->buffer_used = 0;
}

void buf_delete(buffer_manager_t* manager)
{
    free(manager->buffer);
}

typedef struct
{
    int fid;
    int next;
} fid_list_t;

typedef struct {
    int path_plus_1;
    int children_plus_1;
} trie_node_t;

typedef struct {
    int prev_path;
    int index;
    int fid_list;
} path_t;

typedef struct {
    int label_number;
    int root;
    int start_path;
    int path_count;
    int fid_count;
    buffer_manager_t* node_manager;
    buffer_manager_t* path_manager;
    buffer_manager_t* fid_list_manager;
} trie_t;

#define INVALID (-1)
#define EMPTY (0)
#define IS_VALID(x) (x != -1)

#define NODE(trie, node) ((trie_node_t*)buf_from_index((trie)->node_manager, (node)))
#define PATH(trie, path) ((path_t*)buf_from_index((trie)->path_manager, (path)))
#define FID_LIST(trie, fid_list) ((fid_list_t*)buf_from_index((trie)->fid_list_manager, (fid_list)))

#define GET_PATH(trie, node) (NODE((trie), (node))->path_plus_1 - 1)
#define ASSIGN_PATH(trie, node) { if (!IS_VALID(GET_PATH((trie), (node)))) { NODE((trie), (node))->path_plus_1 = buf_get_new_index((trie)->path_manager, 1) + 1; PATH(trie, GET_PATH(trie, node))->fid_list = INVALID; (trie)->path_count++; } }
#define GET_CHILD(trie, node, child_index) (NODE((trie), (node))->children_plus_1 - 1 + child_index)
#define CREATE_CHILDREN(trie, node) (NODE((trie), (node))->children_plus_1 = buf_get_new_index((trie)->node_manager, (trie)->label_number) + 1)
#define HAS_CHILDREN(trie, node) (NODE((trie), (node))->children_plus_1 != 0)

#define CREATE_PATH(trie) (buf_get_new_index((trie)->path_manager, 1))

#define APPEND(trie, fid_list, fid) { int new_fid_list = buf_get_new_index((trie)->fid_list_manager, 1); FID_LIST((trie), new_fid_list)->next = (fid_list); FID_LIST((trie), new_fid_list)->fid = fid; (fid_list) = new_fid_list; }

void trie_init(
    trie_t* trie,
    int label_number,
    buffer_manager_t* node_manager,
    buffer_manager_t* path_manager,
    buffer_manager_t* fid_list_manager
    )
{
    trie->label_number = label_number;
    trie->node_manager = node_manager;
    trie->path_manager = path_manager;
    trie->fid_list_manager = fid_list_manager;
    trie->root = buf_get_new_index(node_manager, 1);
    trie->path_count = 0;
    trie->fid_count = 0;
}

int trie_set_feature(trie_t* trie, crfvol_feature_t* f, int fid, int* created)
{
    int i;
    int node = trie->root;
    int path;

    for (i = 0; i < f->order; i++) {
        if (!HAS_CHILDREN(trie, node)) {
            CREATE_CHILDREN(trie, node);
        }
        node = GET_CHILD(trie, node, f->label_sequence[i]);
    }
    
    path = GET_PATH(trie, node);

    if (IS_VALID(path)) {
        *created = 0;
    } else {
        *created = 1;
        ASSIGN_PATH(trie, node);
        path = GET_PATH(trie, node);
    }

    if (IS_VALID(fid)) {
        APPEND(trie, PATH(trie, path)->fid_list, fid);
        trie->fid_count++;
    }

    return path;
}

int trie_get_longest_match_path_index(trie_t* trie, uint8_t* label_sequence, int label_sequence_len)
{
    int cur_node = trie->root;
    int ret_node = cur_node;
    int i, last_valid_path;

    for (i = 0; i < label_sequence_len; ++i) {
        int path;
        if (!HAS_CHILDREN(trie, cur_node)) break;
        cur_node = GET_CHILD(trie, cur_node, label_sequence[i]);
        path = GET_PATH(trie, cur_node);
        if (IS_VALID(path)) last_valid_path = path;
    }
    return PATH(trie, last_valid_path)->index;
}

int trie_get_path_count(trie_t* trie)
{
    return trie->path_count;
}

typedef struct {
    trie_t* trie;
    trie_t* prev_trie;
    int cur_path_index;
    int cur_fid_index;
    crfvopd_t* preprocessed_data;
} recursion_data_t;

void trie_get_preprocessed_data_(
    int node,
    int valid_parent_index,
    recursion_data_t* r
    )
{
    int path = GET_PATH(r->trie, node);
    if (IS_VALID(path)) {
        int prev_path = PATH(r->trie, path)->prev_path;
        int fid_list = PATH(r->trie, path)->fid_list;

        r->preprocessed_data->paths[r->cur_path_index].longest_suffix_index = valid_parent_index;
        r->preprocessed_data->paths[r->cur_path_index].prev_path_index =
            IS_VALID(prev_path) ? PATH(r->prev_trie, prev_path)->index : INVALID;
        PATH(r->trie, path)->index = r->cur_path_index;

        valid_parent_index = r->cur_path_index;
        r->preprocessed_data->paths[r->cur_path_index].feature_count = 0;

        while (IS_VALID(fid_list)) {
            r->preprocessed_data->fids[r->cur_fid_index] = FID_LIST(r->trie, fid_list)->fid;
            fid_list = FID_LIST(r->trie, fid_list)->next;
            r->preprocessed_data->paths[r->cur_path_index].feature_count++;
            r->cur_fid_index++;
        }
        r->cur_path_index++;
    }
    if (HAS_CHILDREN(r->trie, node)) {
        int i;
        for (i = 0; i < r->trie->label_number; ++i) {
            trie_get_preprocessed_data_(GET_CHILD(r->trie, node, i), valid_parent_index, r);
        }
    }
}

void trie_get_preprocessed_data(
    trie_t* trie,
    trie_t* prev_trie,
    crfvopd_t** preprocessed_data_p,
    uint8_t* label_sequence,
    int label_sequence_len
    )
{
    int i;
    int root = trie->root;
    int path = GET_PATH(trie, root);
    int cur_path_index = 0;
    int valid_parent_index = 0;
    int prev_index_by_label;
    recursion_data_t r;
    crfvopd_t* preprocessed_data;
    
    preprocessed_data = crfvopd_new(trie->label_number, trie->path_count, trie->fid_count);

    /* empty path */
    preprocessed_data->paths[cur_path_index].feature_count = 0;
    preprocessed_data->paths[cur_path_index].longest_suffix_index = INVALID;
    preprocessed_data->paths[cur_path_index].prev_path_index = INVALID;
    valid_parent_index = cur_path_index;
    PATH(trie, 0)->index = cur_path_index;
    cur_path_index++;

    prev_index_by_label = 1;

    r.cur_path_index = cur_path_index;
    r.cur_fid_index = 0;
    r.trie = trie;
    r.prev_trie = prev_trie;
    r.preprocessed_data = preprocessed_data;

    for (i = 0; i < trie->label_number; ++i) {
        trie_get_preprocessed_data_(GET_CHILD(trie, root, i), valid_parent_index, &r);
        preprocessed_data->num_paths_by_label[i] = r.cur_path_index - prev_index_by_label;
        prev_index_by_label = r.cur_path_index;
    }

    preprocessed_data->training_path_index = trie_get_longest_match_path_index(trie, label_sequence, label_sequence_len);

    *preprocessed_data_p = preprocessed_data;
}

crfvopp_t* crfvopp_new()
{
    crfvopp_t* pp = (crfvopp_t*)malloc(sizeof(crfvopp_t));
    if (!pp) return 0;

    pp->path_manager = (buffer_manager_t*)malloc(sizeof(buffer_manager_t));
    pp->node_manager = (buffer_manager_t*)malloc(sizeof(buffer_manager_t));
    pp->fid_list_manager = (buffer_manager_t*)malloc(sizeof(buffer_manager_t));
    if (pp->path_manager && pp->node_manager && pp->fid_list_manager) {
        buf_init(pp->path_manager, sizeof(path_t), 65536);
        buf_init(pp->node_manager, sizeof(trie_node_t), 65536);
        buf_init(pp->fid_list_manager, sizeof(fid_list_t), 65536);
    }
    return pp;
}

void crfvopp_delete(crfvopp_t* pp)
{
    if (pp->path_manager) buf_delete(pp->path_manager);
    if (pp->node_manager) buf_delete(pp->node_manager);
    if (pp->fid_list_manager) buf_delete(pp->fid_list_manager);
    free(pp->path_manager);
    free(pp->node_manager);
    free(pp->fid_list_manager);
    pp->path_manager = pp->node_manager = pp->fid_list_manager = 0;
    free(pp);
    pp = 0;
}

void crfvopp_preprocess_sequence(
    crfvopp_t* pp,
    const feature_refs_t* attrs,
    const crfvol_feature_t* features,
    const int num_labels,
    crf_sequence_t* seq)
{
    const int T = seq->num_items;
    const int L = num_labels;
    int i, j, l, r, t;
    crf_item_t* item;
    trie_t* trie_array;
    trie_t* trie_array_orig;
    uint8_t* label_sequence = malloc(sizeof(uint8_t) * (T+1));

    trie_array_orig = malloc(sizeof(trie_t) * (T+1));
    trie_array = trie_array_orig + 1;
    seq->max_paths = 0;

    for (t = -1; t < T; ++t) { /* -1: BOS */
        int created;
        crfvol_feature_t feature;

        trie_init(&trie_array[t], L+1, pp->node_manager, pp->path_manager, pp->fid_list_manager);

        feature.order = 0;
        trie_set_feature(&trie_array[t], &feature, INVALID, &created);

        if (t == -1 || t == T-1) { /* BOS or EOS */
            feature.label_sequence[0] = L;
            feature.order = 1;
            trie_set_feature(&trie_array[t], &feature, INVALID, &created);
        } else {
            for (l = 0; l < L; ++l) {
                feature.label_sequence[0] = l;
                feature.order = 1;
                trie_set_feature(&trie_array[t], &feature, INVALID, &created);
            }
        }
        if (t == -1) continue; /* BOS */

        item = &seq->items[t];
        
        for (i = 0; i < item->num_contents; ++i) {
            int a = item->contents[i].aid;
            const feature_refs_t* attr = &attrs[a];

            /* Loop over features for the attribute. */
            for (r = 0; r < attr->num_features; ++r) {
                int next_path;
                int fid;
                const crfvol_feature_t* f;

                fid = attr->fids[r];
                f = &features[fid];
                if (
                    (f->order > t+1 && !(f->order == t+2 && f->label_sequence[f->order-1] == L)) ||
                    (f->label_sequence[f->order-1] == L && t != f->order-2 && f->order > 1) ||
                    (t == T-1 && f->label_sequence[0] != L) ||
                    (t != T-1 && f->label_sequence[0] == L)
                    ) continue;

                next_path = INVALID;
                for (j = 0; j < f->order; ++j) {
                    int created;
                    int path;
                    crfvol_feature_t f2;
                    
                    f2.attr = f->attr;
                    f2.order = f->order - j;
                    memcpy(f2.label_sequence, f->label_sequence + j, (MAX_ORDER - j) * sizeof(uint8_t));

                    path = trie_set_feature(&trie_array[t-j], &f2, fid, &created);

                    if (IS_VALID(next_path)) {
                        PATH(&trie_array[t-j+1], next_path)->prev_path = path;
                    }
                    if (j == f->order-1) {
                        int prev = (t-j == -1) ? INVALID : EMPTY;
                        PATH(&trie_array[t-j], path)->prev_path = prev;
                    }
                    if (!created) break;
                    next_path = path;
                    fid = -1;
                }
            }
        }
    }

    label_sequence[T] = L;
    PATH(&(trie_array[-1]), GET_PATH(&(trie_array[-1]), trie_array[-1].root))->index = 0;
    PATH(&(trie_array[-1]), GET_PATH(&(trie_array[-1]), GET_CHILD(&trie_array[-1], trie_array[-1].root, L)))->index = 1;
    for (t = 0; t < T; ++t) {
        item = &seq->items[t];
        label_sequence[T-t-1] = item->label;

        trie_get_preprocessed_data(
            &trie_array[t],
            &trie_array[t-1],
            (crfvopd_t**)&(item->preprocessed_data),
            &(label_sequence[T-t-1]),
            t+2
            );
        item->preprocessed_data_delete_func = crfvopd_delete;
        if (((crfvopd_t*)item->preprocessed_data)->num_paths > seq->max_paths) {
            seq->max_paths = ((crfvopd_t*)item->preprocessed_data)->num_paths;
        }
    }

    buf_clear(pp->node_manager);
    buf_clear(pp->path_manager);
    buf_clear(pp->fid_list_manager);

    free(trie_array_orig);
    free(label_sequence);
}
