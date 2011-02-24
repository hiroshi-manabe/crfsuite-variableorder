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


crfvo_preprocessed_data_t* crfvopp_new(int L, int num_paths, int num_fids)
{
	crfvo_preprocessed_data_t* pp = (crfvo_preprocessed_data_t*)calloc(1, sizeof(crfvo_preprocessed_data_t));	

	pp->num_fids = num_fids;
	pp->num_paths = num_paths;
	pp->fids = (int*)malloc(sizeof(int) * num_fids);
	pp->paths = (crf_path_t*)malloc(sizeof(crf_path_t) * num_paths);
	pp->num_paths_by_label = (int*)malloc(sizeof(int) * (L+1));
	return pp;
}

void crfvopp_delete(crfvo_preprocessed_data_t* pp)
{
	if (pp != NULL) {
		free(pp->fids);
		free(pp->paths);
		free(pp->num_paths_by_label);
	}
	free(pp);
}

