/*
 *      Linear-chain Conditional Random Fields (CRF).
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

/* $Id: crfvo.c 176 2010-07-14 09:31:04Z naoaki $ */

#ifdef    HAVE_CONFIG_H
#include <config.h>
#endif/*HAVE_CONFIG_H*/

#include <os.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <crfsuite.h>
#include "params.h"

#include "logging.h"
#include "crfvo.h"

int crfvo_model_create(const char *filename, crf_model_t** ptr_model);

int crfvo_create_instance_from_file(const char *filename, void **ptr)
{
    return crfvo_model_create(filename, (crf_model_t**)ptr);
}


typedef struct {
    crfvom_t*    crfvom;

    crf_dictionary_t*    attrs;
    crf_dictionary_t*    labels;
    crf_tagger_t*        tagger;
} model_internal_t;

/*
 *    Implementation of crf_dictionary_t object representing attributes.
 *    This object is instantiated only by a crf_model_t object.
 */

static int model_attrs_addref(crf_dictionary_t* dic)
{
    return crf_interlocked_increment(&dic->nref);
}

static int model_attrs_release(crf_dictionary_t* dic)
{
    /* This object is released only by the crf_model_t object. */
    return crf_interlocked_decrement(&dic->nref);
}

static int model_attrs_get(crf_dictionary_t* dic, const char *str)
{
    return CRFERR_NOTSUPPORTED;    /* This object is ready only. */
}

static int model_attrs_to_id(crf_dictionary_t* dic, const char *str)
{
    crfvom_t *crfvom = (crfvom_t*)dic->internal;
    return crfvom_to_aid(crfvom, str);
}

static int model_attrs_to_string(crf_dictionary_t* dic, int id, char const **pstr)
{
    crfvom_t *crfvom = (crfvom_t*)dic->internal;
    const char *str = crfvom_to_attr(crfvom, id);
    *pstr = str;
    return 0;
}

static int model_attrs_num(crf_dictionary_t* dic)
{
    crfvom_t *crfvom = (crfvom_t*)dic->internal;
    return crfvom_get_num_attrs(crfvom);
}

static void model_attrs_free(crf_dictionary_t* dic, const char *str)
{
    /* Unnecessary: all strings are freed on the final release. */
}




/*
 *    Implementation of crf_dictionary_t object representing labels.
 *    This object is instantiated only by a crf_model_t object.
 */

static int model_labels_addref(crf_dictionary_t* dic)
{
    return crf_interlocked_increment(&dic->nref);
}

static int model_labels_release(crf_dictionary_t* dic)
{
    /* This object is released only by the crf_model_t object. */
    return crf_interlocked_decrement(&dic->nref);
}

static int model_labels_get(crf_dictionary_t* dic, const char *str)
{
    return CRFERR_NOTSUPPORTED;    /* This object is ready only. */
}

static int model_labels_to_id(crf_dictionary_t* dic, const char *str)
{
    crfvom_t *crfvom = (crfvom_t*)dic->internal;
    return crfvom_to_lid(crfvom, str);
}

static int model_labels_to_string(crf_dictionary_t* dic, int id, char const **pstr)
{
    crfvom_t *crfvom = (crfvom_t*)dic->internal;
    const char *str = crfvom_to_label(crfvom, id);
    *pstr = str;
    return 0;
}

static int model_labels_num(crf_dictionary_t* dic)
{
    crfvom_t *crfvom = (crfvom_t*)dic->internal;
    return crfvom_get_num_labels(crfvom);
}

static void model_labels_free(crf_dictionary_t* dic, const char *str)
{
    /* Unnecessary: all strings are freed on the final release. */
}



static int tagger_addref(crf_tagger_t* tagger)
{
    return crf_interlocked_increment(&tagger->nref);
}

static int tagger_release(crf_tagger_t* tagger)
{
    /* This object is released only by the crf_model_t object. */
    return crf_interlocked_decrement(&tagger->nref);
}

static int tagger_tag(crf_tagger_t* tagger, crf_sequence_t *inst, crf_output_t* output)
{
    crfvot_t* crfvot = (crfvot_t*)tagger->internal;
    crfvot_tag(crfvot, inst, output);
    return 0;
}

/*
 *    Implementation of crf_model_t object.
 *    This object is instantiated by crfvo_model_create() function.
 */

static int model_addref(crf_model_t* model)
{
    return crf_interlocked_increment(&model->nref);
}

static int model_release(crf_model_t* model)
{
    int count = crf_interlocked_decrement(&model->nref);
    if (count == 0) {
        /* This instance is being destroyed. */
        model_internal_t* internal = (model_internal_t*)model->internal;
        crfvot_delete((crfvot_t*)internal->tagger->internal);
        free(internal->tagger);
        free(internal->labels);
        free(internal->attrs);
        crfvom_close(internal->crfvom);
        free(internal);
        free(model);
    }
    return count;
}

static int model_get_tagger(crf_model_t* model, crf_tagger_t** ptr_tagger)
{
    model_internal_t* internal = (model_internal_t*)model->internal;
    crf_tagger_t* tagger = internal->tagger;

    tagger->addref(tagger);
    *ptr_tagger = tagger;
    return 0;
}

static int model_get_labels(crf_model_t* model, crf_dictionary_t** ptr_labels)
{
    model_internal_t* internal = (model_internal_t*)model->internal;
    crf_dictionary_t* labels = internal->labels;

    labels->addref(labels);
    *ptr_labels = labels;
    return 0;
}

static int model_get_attrs(crf_model_t* model, crf_dictionary_t** ptr_attrs)
{
    model_internal_t* internal = (model_internal_t*)model->internal;
    crf_dictionary_t* attrs = internal->attrs;

    attrs->addref(attrs);
    *ptr_attrs = attrs;
    return 0;
}

static int model_dump(crf_model_t* model, FILE *fpo)
{
    model_internal_t* internal = (model_internal_t*)model->internal;
    crfvom_dump(internal->crfvom, fpo);
    return 0;
}

int crfvo_model_create(const char *filename, crf_model_t** ptr_model)
{
    int ret = 0;
    crfvom_t *crfvom = NULL;
    crfvot_t *crfvot = NULL;
    crf_model_t *model = NULL;
    model_internal_t *internal = NULL;
    crf_tagger_t *tagger = NULL;
    crf_dictionary_t *attrs = NULL, *labels = NULL;

    *ptr_model = NULL;

    /* Open the model file. */
    crfvom = crfvom_new(filename);
    if (crfvom == NULL) {
        ret = CRFERR_INCOMPATIBLE;
        goto error_exit;
    }

    /* Construct a tagger based on the model. */
    crfvot = crfvot_new(crfvom);
    if (crfvot == NULL) {
        ret = CRFERR_OUTOFMEMORY;
        goto error_exit;
    }

    /* Create an instance of internal data attached to the model. */
    internal = (model_internal_t*)calloc(1, sizeof(model_internal_t));
    if (internal == NULL) {
        ret = CRFERR_OUTOFMEMORY;
        goto error_exit;
    }

    /* Create an instance of dictionary object for attributes. */
    attrs = (crf_dictionary_t*)calloc(1, sizeof(crf_dictionary_t));
    if (attrs == NULL) {
        ret = CRFERR_OUTOFMEMORY;
        goto error_exit;
    }
    attrs->internal = crfvom;
    attrs->addref = model_attrs_addref;
    attrs->release = model_attrs_release;
    attrs->get = model_attrs_get;
    attrs->to_id = model_attrs_to_id;
    attrs->to_string = model_attrs_to_string;
    attrs->num = model_attrs_num;
    attrs->free_ = model_attrs_free;

    /* Create an instance of dictionary object for labels. */
    labels = (crf_dictionary_t*)calloc(1, sizeof(crf_dictionary_t));
    if (labels == NULL) {
        ret = CRFERR_OUTOFMEMORY;
        goto error_exit;
    }
    labels->internal = crfvom;
    labels->addref = model_labels_addref;
    labels->release = model_labels_release;
    labels->get = model_labels_get;
    labels->to_id = model_labels_to_id;
    labels->to_string = model_labels_to_string;
    labels->num = model_labels_num;
    labels->free_ = model_labels_free;

    /* */
    /* Create an instance of tagger object. */
    tagger = (crf_tagger_t*)calloc(1, sizeof(crf_tagger_t));
    if (tagger == NULL) {
        ret = CRFERR_OUTOFMEMORY;
        goto error_exit;
    }
    tagger->internal = crfvot;
    tagger->addref = tagger_addref;
    tagger->release = tagger_release;
    tagger->tag = tagger_tag;

    /* Set the internal data. */
    internal->crfvom = crfvom;
    internal->attrs = attrs;
    internal->labels = labels;
    internal->tagger = tagger;

    /* Create an instance of model object. */
    model = (crf_model_t*)calloc(1, sizeof(crf_model_t));
    if (model == NULL) {
        ret = CRFERR_OUTOFMEMORY;
        goto error_exit;
    }
    model->internal = internal;
    model->nref = 1;
    model->addref = model_addref;
    model->release = model_release;
    model->get_attrs = model_get_attrs;
    model->get_labels = model_get_labels;
    model->get_tagger = model_get_tagger;
    model->dump = model_dump;

    *ptr_model = model;
    return 0;

error_exit:
    free(tagger);
    free(labels);
    free(attrs);
    crfvot_delete(crfvot);
    crfvom_close(crfvom);
    free(internal);
    free(model);
    return ret;
}
