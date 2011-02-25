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

struct tag_buffer_manager;
typedef struct tag_buffer_manager buffer_manager_t;

crfvo_preprocessed_data_t* crfvopd_new(int L, int num_paths, int num_fids)
{
	crfvo_preprocessed_data_t* pd = (crfvo_preprocessed_data_t*)calloc(1, sizeof(crfvo_preprocessed_data_t));	

	pd->num_paths = num_paths;
	pd->num_fids = num_fids;
	pd->fids = (int*)malloc(sizeof(int) * num_fids);
	pd->paths = (crfvo_path_t*)malloc(sizeof(crfvo_path_t) * num_paths);
	pd->num_paths_by_label = (int*)malloc(sizeof(int) * (L+1));
	return pd;
}

void crfvopd_delete(crfvo_preprocessed_data_t* pd)
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
	int buffer_size;
};

void buf_init(buffer_manager_t* manager, int unit_size, int initial_buffer_max)
{
	manager->unit_size = unit_size;
	manager->buffer_max = initial_buffer_max;
	manager->buffer = malloc(unit_size * initial_buffer_max);
	memset(manager->buffer, 0, unit_size * initial_buffer_max);
	manager->buffer_size = 0;
}

void* buf_from_index(buffer_manager_t* manager, int index)
{
	return (char*)manager->buffer + manager->unit_size * index;
}

int buf_get_new_index(buffer_manager_t* manager, int count)
{
	int ret;
	while (manager->buffer_max < manager->buffer_size + count) {
		void* temp = realloc(manager->buffer, manager->buffer_max * 2 * manager->unit_size);
		if (!temp) return -1;
		manager->buffer = temp;
		manager->buffer_max *= 2;
		memset((char*)manager->buffer + manager->unit_size * manager->buffer_size, 0,
			manager->unit_size * (manager->buffer_max - manager->buffer_size));
	}
	ret = manager->buffer_size;
	manager->buffer_size += count;
	return ret;
}

int buf_get_current_index(buffer_manager_t* manager)
{
	return manager->buffer_size;
}

void buf_clear(buffer_manager_t* manager)
{
	memset(manager->buffer, 0, manager->unit_size * manager->buffer_size);
	manager->buffer_size = 0;
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
#define IS_VALID(x) (x != -1)

#define NODE(trie, node) ((trie_node_t*)buf_from_index(trie->node_manager, node))
#define GET_PATH(trie, node) (NODE(trie, node)->path_plus_1 - 1)
#define ASSIGN_PATH(trie, node) { if (!IS_VALID(GET_PATH(trie, node))) { NODE(trie, node)->path_plus_1 = buf_get_new_index(trie->path_manager, 1) + 1;  trie->path_count++; } }
#define GET_CHILD(trie, node, child_index) (NODE(trie, node)->children_plus_1 - 1 + child_index)
#define CREATE_CHILDREN(trie, node) (NODE(trie, node)->children_plus_1 = buf_get_new_index(trie->node_manager, trie->label_number))
#define HAS_CHILDREN(trie, node) (NODE(trie, node)->children_plus_1 != 0)

#define PATH(trie, path) ((path_t*)buf_from_index(trie->path_manager, path))
#define CREATE_PATH(trie) (buf_get_new_index(trie->path_manager, 1))

#define FID_LIST(trie, fid_list) ((fid_list_t*)buf_from_index(trie->path_manager, fid_list))
#define APPEND(trie, fid_list, fid) { int new_fid_list = buf_get_new_index(trie->fid_list_manager, 1); FID_LIST(trie, new_fid_list)->next = fid_list; FID_LIST(trie, new_fid_list)->fid = fid; fid_list = new_fid_list; }

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

void trie_insert_path(trie_t* trie, crfvol_feature_t* f, int fid)
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
	
	ASSIGN_PATH(trie, node);
	path = GET_PATH(trie, node);
	if (IS_VALID(fid)) {
		int fid_list = PATH(trie, path)->fid_list;
		APPEND(trie, fid_list, fid);
	}
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
	crfvo_preprocessed_data_t* preprocessed_data;
} recursion_data_t;

void trie_get_preprocessed_data_(int node, int valid_parent_index, recursion_data_t* r)
{
	int path = GET_PATH(r->trie, node);
	if (IS_VALID(path)) {
		int prev_path = PATH(r->trie, node)->prev_path;
		int fid_list = PATH(r->trie, node)->fid_list;

		r->preprocessed_data->paths[r->cur_path_index].longest_suffix_index = valid_parent_index;
		r->preprocessed_data->paths[r->cur_path_index].prev_path_index = PATH(r->prev_trie, prev_path)->index;
		valid_parent_index = r->cur_path_index;

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

void trie_get_preprocessed_data(trie_t* trie, trie_t* prev_trie, crfvo_preprocessed_data_t* preprocessed_data, int* num_paths_by_label)
{
	int i;
	int root = trie->root;
	int path = GET_PATH(trie, root);
	int cur_path_index = 0;
	int valid_parent_index = 0;
	int prev_index_by_label;
	recursion_data_t r;
	
	preprocessed_data->num_paths = trie->path_count;
	preprocessed_data->paths = (crfvo_path_t*)malloc(trie->path_count * sizeof(crfvo_path_t));
	preprocessed_data->num_fids = trie->fid_count;
	preprocessed_data->fids = (int*)malloc(trie->fid_count * sizeof(int));

	/* empty path */
	preprocessed_data->paths[cur_path_index].feature_count = 0;
	preprocessed_data->paths[cur_path_index].longest_suffix_index = INVALID;
	preprocessed_data->paths[cur_path_index].prev_path_index = INVALID;
	valid_parent_index = cur_path_index;
	cur_path_index++;

	prev_index_by_label = 1;

	r.cur_path_index = cur_path_index;
	r.cur_fid_index = 0;
	r.trie = trie;
	r.prev_trie = prev_trie;
	r.preprocessed_data = preprocessed_data;

	for (i = 0; i < trie->label_number; ++i) {
		trie_get_preprocessed_data_(GET_CHILD(trie, root, i), valid_parent_index, &r);
		num_paths_by_label[i] = r.cur_path_index - prev_index_by_label;
		prev_index_by_label = r.cur_path_index;
	}
}

void crfvopp_new(crfvopp_t* pp)
{
	pp->path_manager = (buffer_manager_t*)malloc(sizeof(buffer_manager_t));
}

void crfvopp_delete(crfvopp_t* pp)
{
	buf_delete(pp->path_manager);
}

