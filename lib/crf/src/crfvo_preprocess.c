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


crfvo_preprocessed_data_t* crfvopd_new(int L, int num_paths, int num_fids)
{
	crfvo_preprocessed_data_t* pd = (crfvo_preprocessed_data_t*)calloc(1, sizeof(crfvo_preprocessed_data_t));	

	pd->num_fids = num_fids;
	pd->num_paths = num_paths;
	pd->fids = (int*)malloc(sizeof(int) * num_fids);
	pd->paths = (crf_path_t*)malloc(sizeof(crf_path_t) * num_paths);
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

typedef struct
{
	void* buffer_;
	int unit_size_;
	int buffer_max_;
	int buffer_size_;
}  buffer_manager_t;

void buf_init(buffer_manager_t* manager, int unit_size, int initial_buffer_max)
{
	manager->unit_size_ = unit_size;
	manager->buffer_max_ = initial_buffer_max;
	manager->buffer_ = malloc(unit_size * initial_buffer_max);
	memset(manager->buffer_, 0, unit_size * initial_buffer_max);
	manager->buffer_size_ = 0;
}

void* buf_from_index(buffer_manager_t* manager, int index)
{
	return (char*)manager->buffer_ + manager->unit_size_ * index;
}

int buf_get_new_index(buffer_manager_t* manager, int count)
{
	int ret;
	while (manager->buffer_max_ < manager->buffer_size_ + count) {
		void* temp = realloc(manager->buffer_, manager->buffer_max_ * 2 * manager->unit_size_);
		if (!temp) return -1;
		manager->buffer_ = temp;
		manager->buffer_max_ *= 2;
		memset((char*)manager->buffer_ + manager->unit_size_ * manager->buffer_size_, 0,
			manager->unit_size_ * (manager->buffer_max_ - manager->buffer_size_));
	}
	ret = manager->buffer_size_;
	manager->buffer_size_ += count;
	return ret;
}

int buf_get_current_index(buffer_manager_t* manager)
{
	return manager->buffer_size_;
}

void buf_clear(buffer_manager_t* manager)
{
	memset(manager->buffer_, 0, manager->unit_size_ * manager->buffer_size_);
	manager->buffer_size_ = 0;
}

void buf_free(buffer_manager_t* manager)
{
	free(manager->buffer_);
}

