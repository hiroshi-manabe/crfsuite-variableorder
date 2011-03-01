/*
 *      Linear-chain CRF training.
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

/* $Id: crfvo_learn.c 176 2010-07-14 09:31:04Z naoaki $ */

#include <vector>
#include <list>
#include <memory>
using namespace std;

template<class T> class BufferManager
{
	T* buffer_;
	int buffer_max_;
	int buffer_size_;
public:
	BufferManager(int initial_buffer_max = 65536) {
		buffer_max_ = initial_buffer_max;
		buffer_ = (T*)malloc(sizeof(T) * initial_buffer_max);
		memset(buffer_, 0, sizeof(T) * initial_buffer_max);
		buffer_size_ = 0;
	}

	inline T* from_index(int index) { return buffer_ + index; }

	int get_new_index(int size = 1)
	{
		while (buffer_max_ < buffer_size_ + size) {
			T* temp = (T*)realloc(buffer_, buffer_max_ * 2 * sizeof(T));
			if (!temp) throw;
			buffer_ = temp;
			buffer_max_ *= 2;
			memset(buffer_ + buffer_size_, 0,
				(buffer_max_ - buffer_size_) * sizeof(T));
		}
		int ret = buffer_size_;
		buffer_size_ += size;
		return ret;
	}

	int get_current_index()
	{
		return buffer_size_;
	}

	void clear()
	{
		memset(buffer_, 0, sizeof(T) * buffer_size_);
		buffer_size_ = 0;
	}

	virtual ~BufferManager()
	{
		free(buffer_);
	}
};

#ifdef __cplusplus
extern "C" {
#endif

#ifdef    HAVE_CONFIG_H
#include <config.h>
#endif/*HAVE_CONFIG_H*/

#include <os.h>

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>

#include <crfsuite.h>
#include "params.h"
#include "mt19937ar.h"

#include "logging.h"
#include "crfvo.h"

#define    FEATURE(trainer, k) \
    (&(trainer)->features[(k)])
#define    ATTRIBUTE(trainer, a) \
    (&(trainer)->attributes[(a)])

struct Path : crfvo_path_t
{
	static BufferManager<Path> manager;
};
BufferManager<Path> Path::manager;
#define PATH(index) (Path::manager.from_index(index))

struct FeatureIdList
{
	int feature_id;
	int next_index;
	static BufferManager<FeatureIdList> manager;
};
BufferManager<FeatureIdList> FeatureIdList::manager;
#define FEATURE_ID_LIST(index) (FeatureIdList::manager.from_index(index))

struct TrieNode
{
	int path_id_plus_1;
	int children_index_plus_1;
	inline int get_path_id() { return path_id_plus_1-1; };
	inline void set_path_id(int path_id) { path_id_plus_1 = path_id+1; };
	inline int get_child_index(int child) { return children_index_plus_1-1 + child; };
	inline void set_children_index(int children_index) { children_index_plus_1 = children_index+1; };
	inline bool has_children() { return (children_index_plus_1 != 0); }
	static BufferManager<TrieNode> manager;
};
BufferManager<TrieNode> TrieNode::manager;
#define NODE(index) (TrieNode::manager.from_index(index))

class Trie {
	static int label_number;
	int root_index;
	int start_path_index;
	int last_path_id;

public:
	static void set_label_number(int label_number)
	{
		Trie::label_number = label_number;
	}

	Trie()
	{
		init();
	}

	Trie(const Trie& from) {
		init();
	}

	void init()
	{
		if (label_number == -1) throw;
		root_index = TrieNode::manager.get_new_index();
		last_path_id = -1;
	}

	int set_path(uint8_t* label_sequence, int label_sequence_len, bool& created)
	{
		int node_index = this->root_index;
		for (int i = 0; i < label_sequence_len; i++) {
			if (!NODE(node_index)->has_children()) {
				int new_children_index = TrieNode::manager.get_new_index(label_number);
				NODE(node_index)->set_children_index(new_children_index);
			}
			node_index = NODE(node_index)->get_child_index(label_sequence[i]);
		}
		int path_id = NODE(node_index)->get_path_id();
		if (path_id == -1) {
			last_path_id++;
			NODE(node_index)->set_path_id(last_path_id);
			path_id = last_path_id;
			created = true;
		} else {
			created = false;
		}
		return path_id;
	}

	int get_longest_match_path_id(vector<uint8_t>& label_sequence)
	{
		int cur_node_index = this->root_index;
		int ret_node_index = cur_node_index;

		for (vector<uint8_t>::reverse_iterator it = label_sequence.rbegin(); it != label_sequence.rend(); it++) {
			if (!NODE(cur_node_index)->has_children()) break;
			cur_node_index = NODE(cur_node_index)->get_child_index(*it);
			if (NODE(cur_node_index)->get_path_id() != -1) ret_node_index = cur_node_index;
		}
		return NODE(ret_node_index)->get_path_id();
	}

	int get_path_count() {
		return last_path_id+1;
	}

	void enumerate_path(int* path_id_to_index, int* num_paths_by_label)
	// path_id_to_index must have the size of maximum path_id
	{
		int prev_path_index = -1;
		this->start_path_index = Path::manager.get_current_index();
		int n = this->root_index;
		int path_id = NODE(n)->get_path_id();
		if (path_id == -1) throw;
		int path_index = Path::manager.get_new_index();
		PATH(path_index)->longest_suffix_index = -1;
		int path_index_from_0 = path_index - this->start_path_index;
		path_id_to_index[path_id] = path_index_from_0;
		prev_path_index = path_index;
		if (NODE(n)->has_children() && num_paths_by_label) {
			int prev_path_num = 1; // 1 for the root node
			for (int i = 0; i < label_number; ++i) {
				enumerate_path_(NODE(n)->get_child_index(i), prev_path_index, path_id_to_index);
				int path_num_all = Path::manager.get_current_index() - this->start_path_index;
				num_paths_by_label[i] = path_num_all - prev_path_num;
				prev_path_num = path_num_all;
			}
		}
	}

private:
	void enumerate_path_(int n, int prev_path_index, int* path_id_to_index)
	{
		int path_id = NODE(n)->get_path_id();
		if (path_id != -1) {
			int path_index = Path::manager.get_new_index();
			PATH(path_index)->longest_suffix_index = prev_path_index;
			int path_index_from_0 = path_index - this->start_path_index;
			path_id_to_index[path_id] = path_index_from_0;
			prev_path_index = path_index;
		}
		if (NODE(n)->has_children()) {
			for (int i = 0; i < label_number; ++i) {
				enumerate_path_(NODE(n)->get_child_index(i), prev_path_index, path_id_to_index);
			}
		}
	}
};

int Trie::label_number = -1;

#define FEATURES(index) (FeatureIdList::manager.from_index(index))

struct PrevIdAndFeatureIdList
{
	int prev_id;
	int feature_id_list_index;
	PrevIdAndFeatureIdList(int id) : prev_id(id), feature_id_list_index(-1) {}

	inline void add_feature(int feature_id) {
		int old_index = feature_id_list_index;
		feature_id_list_index = FeatureIdList::manager.get_new_index();
		FEATURES(feature_id_list_index)->feature_id = feature_id;
		FEATURES(feature_id_list_index)->next_index = old_index;
	}
};

void crfvol_set_context(crfvol_t* trainer, const crf_sequence_t* seq)
{
    int i, t;
    crfvo_context_t* ctx = trainer->ctx;
    const crf_item_t* item = NULL;
    const int T = seq->num_items;

    ctx->num_items = T;

    for (t = 0; t < T; ++t) {
        item = &seq->items[t];
        ctx->labels[t] = item->label;
		crfvo_preprocessed_data_t* preprocessed_data = (crfvo_preprocessed_data_t*)item->preprocessed_data;
		memset(ctx->path_scores[t], 0, sizeof(crfvo_path_score_t) * preprocessed_data->num_paths);
		for (i = 0; i < preprocessed_data->num_paths; ++i) {
			ctx->path_scores[t][i].path = preprocessed_data->paths[i];
		}
		ctx->num_paths[t] = preprocessed_data->num_paths;
		ctx->num_paths_by_label[t] = preprocessed_data->num_paths_by_label;
		ctx->fids_refs[t] = preprocessed_data->fids;
		ctx->training_path_indexes[t] = preprocessed_data->training_path_index;
    }
}

void crfvol_preprocess_sequence(crfvol_t* trainer, crf_sequence_t* seq)
{
    int a, i, t, r, fid;
    const floatval_t *fwd = NULL, *bwd = NULL, *state = NULL, *edge = NULL;
    crfvol_feature_t* f = NULL;
    const feature_refs_t* attr = NULL;
    crf_item_t* item = NULL;
    const int T = seq->num_items;
    const int L = trainer->num_labels;

	TrieNode::manager.clear();
	Trie::set_label_number(L+1);
	FeatureIdList::manager.clear();

	vector<Trie> trie_vector(T+1);
	vector<vector<PrevIdAndFeatureIdList> > id_vector_vector(T+1);
	vector<int> feature_counts(T+1);

#define TRIE_VECTOR(i) (trie_vector[i+1])
#define ID_VECTOR_VECTOR(i) (id_vector_vector[i+1])
#define FEATURE_COUNTS(i) (feature_counts[i+1])

	if (!trainer->preprocessor) {
		trainer->preprocessor = malloc(sizeof(crfvopp_t));
		trainer->preprocessor_delete_func = (void (*)(void*))crfvopp_delete;
	}

	for (t = -1; t < T; ++t) { // -1: BOS
		bool created;

		TRIE_VECTOR(t).set_path(0, 0, created); // id for an empty path is 0
		ID_VECTOR_VECTOR(t).push_back(PrevIdAndFeatureIdList(-1));

		if (t == -1 || t == T-1) { // BOS or EOS
			uint8_t L2 = L;
			TRIE_VECTOR(t).set_path(&L2, 1, created);
			ID_VECTOR_VECTOR(t).push_back(PrevIdAndFeatureIdList(0)); // empty path
		} else {
			for (int l = 0; l < L; ++l) {
				uint8_t L2 = l;
				TRIE_VECTOR(t).set_path(&L2, 1, created);
				ID_VECTOR_VECTOR(t).push_back(PrevIdAndFeatureIdList(0)); // empty path
			}
		}
		if (t == -1) continue; // BOS

        item = &seq->items[t];
		
		for (i = 0; i < item->num_contents; ++i) {
			a = item->contents[i].aid;
			attr = ATTRIBUTE(trainer, a);

			/* Loop over features for the attribute. */
			for (r = 0;r < attr->num_features;++r) {
				fid = attr->fids[r];
				f = FEATURE(trainer, fid);
				if (
					(f->order > t+1 && !(f->order == t+2 && f->label_sequence[f->order-1] == L)) ||
					(f->label_sequence[f->order-1] == L && t != f->order-2 && f->order > 1) ||
					(t == T-1 && f->label_sequence[0] != L) ||
					(t != T-1 && f->label_sequence[0] == L)
					) continue;

				FEATURE_COUNTS(t)++;
				int last_path_id = -1;
				for (int k = 0; k < f->order; ++k) {
					bool created;
					int path_id = TRIE_VECTOR(t - k).set_path(f->label_sequence + k, f->order - k, created);
					if (created) ID_VECTOR_VECTOR(t - k).push_back(PrevIdAndFeatureIdList(-1));
					if (k == 0) {
						ID_VECTOR_VECTOR(t - k)[path_id].add_feature(fid);
					}
					if (last_path_id != -1) ID_VECTOR_VECTOR(t - k + 1)[last_path_id].prev_id = path_id;
					if (k == f->order-1) {
						int prev_id = (t - k == -1) ? -1 : 0;
						ID_VECTOR_VECTOR(t - k)[path_id].prev_id = prev_id; // 0 for the empty path
					}
					if (!created) break;
					last_path_id = path_id;
					fid = -1;
				}
			}
		}
	}

	int max_path_count = 0;
	int max_feature_count = 0;
	for (t = -1; t < T; ++t) {
		int path_count = TRIE_VECTOR(t).get_path_count();
		if (path_count > max_path_count) max_path_count = path_count;
		if (FEATURE_COUNTS(t) > max_feature_count) max_feature_count = FEATURE_COUNTS(t);
	}

	vector<uint8_t> train_label_vector(0);
#define TRAIN_LABEL_VECTOR(i) (train_label_vector[i+1])

	int empty_path_index = 0;
	int* cur_path_id_to_index = (int*)malloc(sizeof(int) * max_path_count);
	int* prev_path_id_to_index = (int*)malloc(sizeof(int) * max_path_count);
	int* path_index_to_id = (int*)malloc(sizeof(int) * max_path_count);
	int* feature_ids = (int*)malloc(sizeof(int) * max_feature_count);

	for (t = -1; t < T; ++t) { // -1: BOS
		Path::manager.clear();

		int label = (t == -1 || t == T-1) ? L : seq->items[t].label;
		train_label_vector.push_back(label);
		int path_count = TRIE_VECTOR(t).get_path_count();
		swap(prev_path_id_to_index, cur_path_id_to_index);

		int* num_paths_by_label = (int*)malloc(sizeof(int) * (L+1));
		TRIE_VECTOR(t).enumerate_path(cur_path_id_to_index, num_paths_by_label);

		for (int id = 0; id < path_count; ++id) {
			path_index_to_id[cur_path_id_to_index[id]] = id;
		}

		int feature_count_all = 0;
		for (int index = 0; index < path_count; ++index) {
			int prev_id = ID_VECTOR_VECTOR(t)[path_index_to_id[index]].prev_id;
			PATH(index)->prev_path_index = 
				(prev_id == -1) ? -1 : prev_path_id_to_index[prev_id];
			int feature_id_list_index = ID_VECTOR_VECTOR(t)[path_index_to_id[index]].feature_id_list_index;
			int feature_count = 0;
			while (feature_id_list_index != -1) {
				feature_ids[feature_count_all++] = FEATURE_ID_LIST(feature_id_list_index)->feature_id;
				feature_count++;
				feature_id_list_index = FEATURE_ID_LIST(feature_id_list_index)->next_index;
			}
			PATH(index)->feature_count = feature_count;
		}
		if (t >= 0) {
			crfvo_preprocessed_data_t* preprocessed_data;
	        item = &seq->items[t];

			if (item->preprocessed_data_delete_func) item->preprocessed_data_delete_func(item->preprocessed_data);
			item->preprocessed_data = (crfvo_preprocessed_data_t*)crfvopd_new(L, path_count, FEATURE_COUNTS(t));
			item->preprocessed_data_delete_func = (void (*)(void*))crfvopd_delete;
			preprocessed_data = (crfvo_preprocessed_data_t*)item->preprocessed_data;

			memcpy(preprocessed_data->fids, feature_ids, FEATURE_COUNTS(t) * sizeof(int));
			memcpy(preprocessed_data->paths, Path::manager.from_index(0), path_count * sizeof(crfvo_path_t));
			preprocessed_data->training_path_index = cur_path_id_to_index[TRIE_VECTOR(t).get_longest_match_path_id(train_label_vector)];
			memcpy(preprocessed_data->num_paths_by_label, num_paths_by_label, sizeof(int) * (L+1));
			t = t + 0;
		}
	}
	free(cur_path_id_to_index);
	free(prev_path_id_to_index);
	free(path_index_to_id);
	free(feature_ids);
	seq->max_paths = max_path_count;
}

void crfvol_enum_features(crfvol_t* trainer, const crf_sequence_t* seq, update_feature_t func, double* logp)
{
    crfvo_context_t* ctx = trainer->ctx;
    const int T = seq->num_items;
    const int L = trainer->num_labels;

	for (int t = 0; t < T; ++t) {
		crfvo_path_score_t* path_scores = ctx->path_scores[t];
		int* fids = ctx->fids_refs[t];
		int n = ctx->num_paths[t];
		int fid_counter = 0;
		for (int i = 0; i < n; ++i) {
			int fid_num = path_scores[i].path.feature_count;
			for (int j = 0; j < fid_num; ++j) {
				floatval_t prob = path_scores[i].score;
				int fid = fids[fid_counter];
				fid_counter++;
				crfvol_feature_t* f = FEATURE(trainer, fid);
				func(f, fid, prob, 1.0, trainer, seq, t);				
			}
		}
	}
}

static int init_feature_references(crfvol_t* trainer, const int A, const int L)
{
    int i, k;
    feature_refs_t *fl = NULL;
    const int K = trainer->num_features;
    const crfvol_feature_t* features = trainer->features;

    /*
        The purpose of this routine is to collect references (indices) of:
        - features fired by each attribute (trainer->attributes)
    */

    /* Initialize. */
    trainer->attributes = NULL;

    /* Allocate arrays for feature references. */
    trainer->attributes = (feature_refs_t*)calloc(A, sizeof(feature_refs_t));
    if (trainer->attributes == NULL) goto error_exit;

    /*
        Firstly, loop over the features to count the number of references.
        We don't want to use realloc() to avoid memory fragmentation.
     */
    for (k = 0;k < K;++k) {
        const crfvol_feature_t *f = &features[k];
        trainer->attributes[f->attr].num_features++;
    }

    /*
        Secondarily, allocate memory blocks to store the feature references.
        We also clear fl->num_features fields, which will be used to indicate
        the offset positions in the last phase.
     */
    for (i = 0;i < trainer->num_attributes;++i) {
        fl = &trainer->attributes[i];
        fl->fids = (int*)calloc(fl->num_features, sizeof(int));
        if (fl->fids == NULL) goto error_exit;
        fl->num_features = 0;
    }

    /*
        At last, store the feature indices.
     */
    for (k = 0;k < K;++k) {
        const crfvol_feature_t *f = &features[k];
        fl = &trainer->attributes[f->attr];
        fl->fids[fl->num_features++] = k;
    }

    return 0;

error_exit:
    if (trainer->attributes == NULL) {
        for (i = 0;i < A;++i) free(trainer->attributes[i].fids);
        free(trainer->attributes);
        trainer->attributes = NULL;
    }
    return -1;
}

int crfvol_prepare(
    crfvol_t* trainer,
    int num_labels,
    int num_attributes,
    int max_item_length,
	int max_paths,
    crfvol_features_t* features
    )
{
    int ret = 0;
    const int L = num_labels;
    const int A = num_attributes;
    const int T = max_item_length;

    /* Set basic parameters. */
    trainer->num_labels = L;
    trainer->num_attributes = A;

    /* Construct a CRF context. */
    trainer->ctx = crfvoc_new(L, T, max_paths);
    if (trainer->ctx == NULL) {
        ret = CRFERR_OUTOFMEMORY;
        goto error_exit;
    }

    /* Initialization for features and their weights. */
    trainer->features = features->features;
    trainer->num_features = features->num_features;
    trainer->w = (floatval_t*)calloc(trainer->num_features, sizeof(floatval_t));
    if (trainer->w == NULL) {
        ret = CRFERR_OUTOFMEMORY;
        goto error_exit;
    }

    /* Allocate the work space for probability calculation. */
    trainer->prob = (floatval_t*)calloc(L, sizeof(floatval_t));
    if (trainer->prob == NULL) {
        ret = CRFERR_OUTOFMEMORY;
        goto error_exit;
    }

    /* Initialize the feature references. */
    init_feature_references(trainer, A, L);

	return ret;

error_exit:
    free(trainer->attributes);
    free(trainer->prob);
    free(trainer->ctx);
    return 0;
}

void crfvol_preprocess(
    crfvol_t* trainer
	)
{
	int i;
	logging(trainer->lg, "Compiling...\n");
	logging_progress_start(trainer->lg);

	for (i = 0; i < trainer->num_sequences; ++i) {
		crfvopp_preprocess_sequence((crfvopp_t*)trainer->preprocessor, trainer, &trainer->seqs[i]);

		if (trainer->max_paths < trainer->seqs[i].max_paths) {
			trainer->max_paths = trainer->seqs[i].max_paths;
		}
		logging_progress(trainer->lg, (100 * i) / trainer->num_sequences);
	}

	logging_progress_end(trainer->lg);
}

int crfvol_set_feature_freqs(
	crfvol_t* trainer,
	crfvol_features_t* features
	)
{
	int i, t, n, ret;
	int* feature_freqs = (int*)calloc(trainer->num_features, sizeof(int));
	int* feature_last_indexes = (int*)malloc(trainer->max_paths * sizeof(int));

	for (i = 0; i < trainer->num_sequences; ++i) {
		crf_sequence_t* seq = &trainer->seqs[i];
		for (t = 0; t < seq->num_items; ++t) {
			crf_item_t* item = &seq->items[t];
			crfvo_preprocessed_data_t* preprocessed_data = (crfvo_preprocessed_data_t*)item->preprocessed_data;
			int feature_last_index = 0;
			for (n = 0; n < preprocessed_data->num_paths; ++n) {
				feature_last_index += preprocessed_data->paths[n].feature_count;
				feature_last_indexes[n] = feature_last_index;
			}
			n = preprocessed_data->training_path_index;
			while (n > 0) {
				int j;
				for (j = feature_last_indexes[n-1]; j < feature_last_indexes[n]; ++j) {
					feature_freqs[preprocessed_data->fids[j]]++;
				}
				n = preprocessed_data->paths[n].longest_suffix_index;
			}
		}
	}
	ret = 1;
	for (i = 0; i < trainer->num_features; ++i) {
		if (trainer->features[i].freq != feature_freqs[i]) {
			trainer->features[i].freq = feature_freqs[i];
		}
	}
	free(feature_freqs);
	free(feature_last_indexes);
	return ret;
}

static int crfvol_exchange_options(crf_params_t* params, crfvol_option_t* opt, int mode)
{
    BEGIN_PARAM_MAP(params, mode)
        DDX_PARAM_STRING(
            "algorithm", opt->algorithm, "lbfgs",
            "The training algorithm."
            )
        DDX_PARAM_FLOAT(
            "feature.minfreq", opt->feature_minfreq, 0.0,
            "The minimum frequency of features."
            )
        DDX_PARAM_INT(
            "feature.emulate_crfvo", opt->feature_emulate_crfvo, 0,
            "Emulate First-order CRF."
            )
    END_PARAM_MAP()

    crfvol_lbfgs_options(params, opt, mode);

    return 0;
}

void crfvol_shuffle(int *perm, int N, int init)
{
    int i, j, tmp;

    if (init) {
        /* Initialize the permutation if necessary. */
        for (i = 0;i < N;++i) {
            perm[i] = i;
        }
    }

    for (i = 0;i < N;++i) {
        j = mt_genrand_int31() % N;
        tmp = perm[j];
        perm[j] = perm[i];
        perm[i] = tmp;
    }
}

crfvol_t* crfvol_new()
{
    crfvol_t* trainer = (crfvol_t*)calloc(1, sizeof(crfvol_t));
    trainer->lg = (logging_t*)calloc(1, sizeof(logging_t));
	trainer->exp_weight = 0;
	trainer->preprocessor = 0;

    /* Create an instance for CRF parameters. */
    trainer->params = params_create_instance();
    /* Set the default parameters. */
    crfvol_exchange_options(trainer->params, &trainer->opt, 0);

    return trainer;
}

void crfvol_delete(crfvol_t* trainer)
{
    if (trainer != NULL) {
        free(trainer->lg);
		free(trainer->exp_weight);
		if (trainer->preprocessor_delete_func) {
			trainer->preprocessor_delete_func(trainer->preprocessor);
		}
		free(trainer->preprocessor);
    }
	free(trainer);
}

int crf_train_tag(crf_tagger_t* tagger, crf_sequence_t *inst, crf_output_t* output)
{
    int i;
    floatval_t logscore = 0;
    crfvol_t *crfvot = (crfvol_t*)tagger->internal;
    const floatval_t* exp_weight = crfvot->exp_weight;
    const int K = crfvot->num_features;
    crfvo_context_t* ctx = crfvot->ctx;
	int max_path = 0;

	for (i = 0; i < inst->num_items; ++i) {
		if (inst->items[i].preprocessed_data == 0) {
			crfvopp_preprocess_sequence((crfvopp_t*)((crfvol_t*)tagger->internal)->preprocessor, (crfvol_t*)tagger->internal, inst);
			break;
		}
	}
    crfvoc_set_num_items(ctx, inst->num_items, inst->max_paths);

    crfvol_set_context(crfvot, inst);
	crfvoc_set_weight(ctx, exp_weight);
    logscore = crfvoc_viterbi(crfvot->ctx);

    crf_output_init_n(output, inst->num_items);
    output->probability = logscore;
    for (i = 0;i < inst->num_items;++i) {
        output->labels[i] = crfvot->ctx->labels[i];
    }
    output->num_labels = inst->num_items;

    return 0;
}

void crf_train_set_message_callback(crf_trainer_t* trainer, void *instance, crf_logging_callback cbm)
{
    crfvol_t *crfvot = (crfvol_t*)trainer->internal;
    crfvot->lg->func = cbm;
    crfvot->lg->instance = instance;
}

void crf_train_set_evaluate_callback(crf_trainer_t* trainer, void *instance, crf_evaluate_callback cbe)
{
    crfvol_t *crfvot = (crfvol_t*)trainer->internal;
    crfvot->cbe_instance = instance;
    crfvot->cbe_proc = cbe;
}

static int crf_train_train(
    crf_trainer_t* trainer,
    void* instances,
    int num_instances,
    int num_labels,
    int num_attributes
    )
{
    int i, max_item_length;
    int ret = 0;
    floatval_t sigma = 10, *best_w = NULL;
    crf_sequence_t* seqs = (crf_sequence_t*)instances;
    crfvol_features_t* features = NULL;
    crfvol_t *crfvot = (crfvol_t*)trainer->internal;
    crf_params_t *params = crfvot->params;
    crfvol_option_t *opt = &crfvot->opt;

    /* Obtain the maximum number of items. */
    max_item_length = 0;
    for (i = 0;i < num_instances;++i) {
        if (max_item_length < seqs[i].num_items) {
            max_item_length = seqs[i].num_items;
        }
    }

    /* Access parameters. */
    crfvol_exchange_options(crfvot->params, opt, -1);

    /* Report the parameters. */
    logging(crfvot->lg, "Training first-order linear-chain CRFs (trainer.crfvo)\n");
    logging(crfvot->lg, "\n");

    /* Generate features. */
    logging(crfvot->lg, "Feature generation\n");
    logging(crfvot->lg, "feature.minfreq: %f\n", opt->feature_minfreq);
    logging(crfvot->lg, "feature.emulate_crfvo: %d\n", opt->feature_emulate_crfvo);
    crfvot->clk_begin = clock();
    features = crfvol_generate_features(
        seqs,
        num_instances,
        num_labels,
        num_attributes,
		opt->feature_emulate_crfvo, // emulate crfvo
        opt->feature_minfreq,
        crfvot->lg->func,
        crfvot->lg->instance
        );
    logging(crfvot->lg, "Number of features: %d\n", features->num_features);
    logging(crfvot->lg, "Seconds required: %.3f\n", (clock() - crfvot->clk_begin) / (double)CLOCKS_PER_SEC);
    logging(crfvot->lg, "\n");

    /* Preparation for training. */
	crfvol_prepare(crfvot, num_labels, num_attributes, max_item_length, 0, features);
    crfvot->num_attributes = num_attributes;
    crfvot->num_labels = num_labels;
    crfvot->num_sequences = num_instances;
    crfvot->seqs = seqs;

	crfvot->preprocessor = malloc(sizeof(crfvopp_t));
	crfvot->preprocessor_delete_func = (void (*)(void*))crfvopp_delete;
	crfvopp_new((crfvopp_t*)crfvot->preprocessor);

	// preprocess
	crfvol_preprocess(crfvot);
	crfvol_set_feature_freqs(crfvot, features);

	crfvoc_set_num_items(crfvot->ctx, max_item_length, crfvot->max_paths);

    crfvot->tagger.internal = crfvot;
    crfvot->tagger.tag = crf_train_tag;

    if (strcmp(opt->algorithm, "lbfgs") == 0) {
        ret = crfvol_lbfgs(crfvot, opt);
    } else {
        return CRFERR_INTERNAL_LOGIC;
    }

    return ret;
}

/*#define    CRF_TRAIN_SAVE_NO_PRUNING    1*/

static int crf_train_save(crf_trainer_t* trainer, const char *filename, crf_dictionary_t* attrs, crf_dictionary_t* labels)
{
    crfvol_t *crfvot = (crfvol_t*)trainer->internal;
    int a, k, l, ret;
    int *fmap = NULL, *amap = NULL;
    crfvomw_t* writer = NULL;
    const feature_refs_t *edge = NULL, *attr = NULL;
    const floatval_t *w = crfvot->w;
    const floatval_t threshold = 0.01;
    const int L = crfvot->num_labels;
    const int A = crfvot->num_attributes;
    const int K = crfvot->num_features;
    int J = 0, B = 0;

    /* Start storing the model. */
    logging(crfvot->lg, "Storing the model\n");
    crfvot->clk_begin = clock();

    /* Allocate and initialize the feature mapping. */
    fmap = (int*)calloc(K, sizeof(int));
    if (fmap == NULL) {
        goto error_exit;
    }
#ifdef    CRF_TRAIN_SAVE_NO_PRUNING
    for (k = 0;k < K;++k) fmap[k] = k;
    J = K;
#else
    for (k = 0;k < K;++k) fmap[k] = -1;
#endif/*CRF_TRAIN_SAVE_NO_PRUNING*/

    /* Allocate and initialize the attribute mapping. */
    amap = (int*)calloc(A, sizeof(int));
    if (amap == NULL) {
        goto error_exit;
    }
#ifdef    CRF_TRAIN_SAVE_NO_PRUNING
    for (a = 0;a < A;++a) amap[a] = a;
    B = A;
#else
    for (a = 0;a < A;++a) amap[a] = -1;
#endif/*CRF_TRAIN_SAVE_NO_PRUNING*/

    /*
     *    Open a model writer.
     */
    writer = crfvomw(filename);
    if (writer == NULL) {
        goto error_exit;
    }

    /* Open a feature chunk in the model file. */
    if (ret = crfvomw_open_features(writer)) {
        goto error_exit;
    }

    /* Determine a set of active features and attributes. */
    for (k = 0;k < crfvot->num_features;++k) {
        crfvol_feature_t* f = &crfvot->features[k];
        if (w[k] != 0) {
            int attr;
            crfvom_feature_t feat;

#ifndef    CRF_TRAIN_SAVE_NO_PRUNING
            /* The feature (#k) will have a new feature id (#J). */
            fmap[k] = J++;        /* Feature #k -> #fmap[k]. */

            /* Map the source of the field. */
            /* The attribute #(f->src) will have a new attribute id (#B). */
            if (amap[f->attr] < 0) amap[f->attr] = B++;    /* Attribute #a -> #amap[a]. */
            attr = amap[f->attr];
#endif/*CRF_TRAIN_SAVE_NO_PRUNING*/

            feat.order = f->order;
            feat.attr = attr;
            memcpy(feat.label_sequence, f->label_sequence, sizeof(feat.label_sequence[0]) * MAX_ORDER);
            feat.weight = w[k];

            /* Write the feature. */
            if (ret = crfvomw_put_feature(writer, fmap[k], &feat)) {
                goto error_exit;
            }
        }
    }

    /* Close the feature chunk. */
    if (ret = crfvomw_close_features(writer)) {
        goto error_exit;
    }

    logging(crfvot->lg, "Number of active features: %d (%d)\n", J, K);
    logging(crfvot->lg, "Number of active attributes: %d (%d)\n", B, A);
    logging(crfvot->lg, "Number of active labels: %d (%d)\n", L, L);

    /* Write labels. */
    logging(crfvot->lg, "Writing labels\n", L);
    if (ret = crfvomw_open_labels(writer, L)) {
        goto error_exit;
    }
    for (l = 0;l < L;++l) {
        const char *str = NULL;
        labels->to_string(labels, l, &str);
        if (str != NULL) {
            if (ret = crfvomw_put_label(writer, l, str)) {
                goto error_exit;
            }
            labels->free_(labels, str);
        }
    }
    if (ret = crfvomw_close_labels(writer)) {
        goto error_exit;
    }

    /* Write attributes. */
    logging(crfvot->lg, "Writing attributes\n");
    if (ret = crfvomw_open_attrs(writer, B)) {
        goto error_exit;
    }
    for (a = 0;a < A;++a) {
        if (0 <= amap[a]) {
            const char *str = NULL;
            attrs->to_string(attrs, a, &str);
            if (str != NULL) {
                if (ret = crfvomw_put_attr(writer, amap[a], str)) {
                    goto error_exit;
                }
                attrs->free_(attrs, str);
            }
        }
    }
    if (ret = crfvomw_close_attrs(writer)) {
        goto error_exit;
    }

    /* Close the writer. */
    crfvomw_close(writer);
    logging(crfvot->lg, "Seconds required: %.3f\n", (clock() - crfvot->clk_begin) / (double)CLOCKS_PER_SEC);
    logging(crfvot->lg, "\n");

    free(amap);
    free(fmap);
    return 0;

error_exit:
    if (writer != NULL) {
        crfvomw_close(writer);
    }
    if (amap != NULL) {
        free(amap);
    }
    if (fmap != NULL) {
        free(fmap);
    }
    return ret;
}

static int crf_train_addref(crf_trainer_t* trainer)
{
    return crf_interlocked_increment(&trainer->nref);
}

static int crf_train_release(crf_trainer_t* trainer)
{
    int count = crf_interlocked_decrement(&trainer->nref);
    if (count == 0) {
		crfvol_t* crfvot = (crfvol_t*)trainer->internal;
		if (crfvot->preprocessor) {
			crfvot->preprocessor_delete_func(crfvot->preprocessor);
			free(crfvot->preprocessor);
		}
    }
    return count;
}

static crf_params_t* crf_train_params(crf_trainer_t* trainer)
{
    crfvol_t *crfvot = (crfvol_t*)trainer->internal;
    crf_params_t* params = crfvot->params;
    params->addref(params);
    return params;
}


int crfvol_create_instance(const char *interface, void **ptr)
{
    if (strcmp(interface, "trainer.crfvo") == 0) {
        crf_trainer_t* trainer = (crf_trainer_t*)calloc(1, sizeof(crf_trainer_t));

        trainer->nref = 1;
        trainer->addref = crf_train_addref;
        trainer->release = crf_train_release;

        trainer->params = crf_train_params;
    
        trainer->set_message_callback = crf_train_set_message_callback;
        trainer->set_evaluate_callback = crf_train_set_evaluate_callback;
        trainer->train = crf_train_train;
        trainer->save = crf_train_save;
        trainer->internal = crfvol_new();

        *ptr = trainer;
        return 0;
    } else {
        return 1;
    }
}
#ifdef __cplusplus
} // extern "C"
#endif
