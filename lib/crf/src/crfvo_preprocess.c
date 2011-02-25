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

	pd->num_fids = num_fids;
	pd->num_paths = num_paths;
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
	int feature_id;
	int next;
} feature_id_list_t;

typedef struct {
	int path_id_plus_1;
	int children_plus_1;
} trie_node_t;

typedef struct {
	int label_number;
	int root;
	int start_path;
	int last_path_id;
	buffer_manager_t* node_manager;
	buffer_manager_t* path_manager;
} trie_t;

#define NODE(trie, node) ((trie_node_t*)buf_from_index(trie->node_manager, node))

#define GET_PATH_ID(trie, node) (NODE(trie, node)->path_id_plus_1 - 1)
#define SET_PATH_ID(trie, node, path_id) (NODE(trie, node)->path_id_plus_1 = path_id + 1)
#define GET_CHILD(trie, node, child_index) (NODE(trie, node)->children_plus_1 - 1 + child_index)
#define CREATE_CHILDREN(trie, node) (NODE(trie, node)->children_plus_1 = buf_get_new_index(trie->node_manager, trie->label_number))
#define HAS_CHILDREN(trie, node) (NODE(trie, node)->children_plus_1 != 0)

#define PATH(trie, path) ((crfvo_path_t*)buf_from_index(trie->path_manager, path))
#define CREATE_PATH(trie) (buf_get_new_index(trie->path_manager, 1))

void trie_init(trie_t* trie, int label_number, buffer_manager_t* node_manager, buffer_manager_t* path_manager)
{
	trie->label_number = label_number;
	trie->node_manager = node_manager;
	trie->path_manager = path_manager;
	trie->root = buf_get_new_index(node_manager, 1);
	trie->last_path_id = -1;
}

int trie_insert_path(trie_t* trie, uint8_t* label_sequence, int label_sequence_len, int* created)
{
	int i;
	int node = trie->root;
	int path_id;

	for (i = 0; i < label_sequence_len; i++) {
		if (!HAS_CHILDREN(trie, node)) {
			CREATE_CHILDREN(trie, node);
		}
		node = GET_CHILD(trie, node, label_sequence[i]);
	}
	
	path_id = GET_PATH_ID(trie, node);

	if (path_id == -1) {
		trie->last_path_id++;
		SET_PATH_ID(trie, node, trie->last_path_id);
		path_id = trie->last_path_id;
		*created = 1;
	} else {
		*created = 0;
	}
	return path_id;
}

int trie_get_longest_match_path_id(trie_t* trie, uint8_t* label_sequence, int label_sequence_len)
{
	int cur_node = trie->root;
	int ret_node = cur_node;
	int i;

	for (i = 0; i < label_sequence_len; ++i) {
		if (!HAS_CHILDREN(trie, cur_node)) break;
		cur_node = GET_CHILD(trie, cur_node, label_sequence[i]);
		if (GET_PATH_ID(trie, cur_node) != -1) ret_node = cur_node;
	}
	return GET_PATH_ID(trie, ret_node);
}

int trie_get_path_count(trie_t* trie)
{
	return trie->last_path_id + 1;
}

void trie_enumerate_path_(trie_t* trie, int node, int valid_parent, int* path_id_to_index)
{
	int path_id = GET_PATH_ID(trie, node);
	if (path_id != -1) {
		int path = buf_get_new_index(trie->path_manager, 1);
		PATH(trie, path)->longest_suffix_index = valid_parent;
		path_id_to_index[path_id] = path - trie->start_path;
		valid_parent = path;
	}
	if (HAS_CHILDREN(trie, node)) {
		int i;
		for (i = 0; i < trie->label_number; ++i) {
			trie_enumerate_path_(trie, GET_CHILD(trie, node, i), valid_parent, path_id_to_index);
		}
	}
}

// path_id_to_index must have the size of maximum path_id
void enumerate_path(trie_t* trie, int* path_id_to_index, int* num_paths_by_label)
{
	int i;
	int root_node = trie->root;
	int valid_parent = -1;
	int path_id = GET_PATH_ID(trie, root_node);
	int root_path;

	trie->start_path = buf_get_current_index(trie->path_manager);
	root_path = CREATE_PATH(trie);
	path_id_to_index[path_id] = root_path - trie->start_path;

	trie_enumerate_path_(trie, root_node, root_path, path_id_to_index);

	for (i = 0; i < trie->label_number; ++i) {
		num_paths_by_label[i] = path_id_to_index[GET_PATH_ID(trie, GET_CHILD(trie, root_node, i))];
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

