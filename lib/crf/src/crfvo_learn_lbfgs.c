/*
 *      Training linear-chain CRF with L-BFGS.
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

/* $Id: crfvo_learn_lbfgs.c 176 2010-07-14 09:31:04Z naoaki $ */

#ifdef    HAVE_CONFIG_H
#include <config.h>
#endif/*HAVE_CONFIG_H*/

#include <os.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>

#include <crfsuite.h>
#include "crfvo.h"

#include "logging.h"
#include "params.h"
#include <lbfgs.h>

#include <math.h>

typedef struct {
    int l2_regularization;
    floatval_t sigma2inv;
    floatval_t* g;
    floatval_t* best_w;
} lbfgs_internal_t;

#define LBFGS_INTERNAL(crfvol)    ((lbfgs_internal_t*)((crfvol)->solver_data))

inline static void update_model_expectations(
    crfvol_feature_t* f,
    const int fid,
    floatval_t prob,
    floatval_t scale,
    crfvol_t* crfvol,
    const crf_sequence_t* seq,
    int t
    )
{
    lbfgs_internal_t *lbfgsi = LBFGS_INTERNAL(crfvol);
    lbfgsi->g[fid] += prob * scale;
}

static lbfgsfloatval_t lbfgs_evaluate(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    )
{
    int i;
    floatval_t logp = 0, logl = 0, norm = 0;
    crfvol_t* crfvot = (crfvol_t*)instance;
    crf_sequence_t* seqs = crfvot->seqs;
    const int N = crfvot->num_sequences;

    lbfgs_internal_t *lbfgsi = LBFGS_INTERNAL(crfvot);

    if (!crfvot->exp_weight) {
        crfvot->exp_weight = (floatval_t*)calloc(crfvot->num_features, sizeof(floatval_t));
    }

    for (i = 0; i < crfvot->num_features; ++i) {
        crfvot->exp_weight[i] = exp(x[i]);
    }

    /* Set the gradient vector. */
    lbfgsi->g = g;

    /*
        Set feature weights from the L-BFGS solver. Initialize model
        expectations as zero.
     */
    for (i = 0;i < crfvot->num_features;++i) {
        crfvol_feature_t* f = &crfvot->features[i];
        g[i] = -f->freq;
    }

    /*
        Compute model expectations.
     */
    for (i = 0;i < N;++i) {
        /* Set label sequences and state scores. */
        crfvoc_set_context(crfvot->ctx, &seqs[i]);

        /*crfvoc_debug_context(crfvot->ctx, stdout);*/
        /*printf("lognorm = %f\n", crfvot->ctx->log_norm);*/

        crfvoc_set_weight(crfvot->ctx, crfvot->exp_weight);
        crfvoc_calc_feature_expectations(crfvot->ctx);

        /* Compute the probability of the input sequence on the model. */
        logp = crfvoc_logprob(crfvot->ctx);
        /* Update the log-likelihood. */
        logl += logp;

        /* Update the model expectations of features. */
        crfvol_enum_features(crfvot, &seqs[i], update_model_expectations, &logp);
    }

    /*
        L2 regularization.
        Note that we *add* the (w * sigma) to g[i].
     */
    if (lbfgsi->l2_regularization) {
        for (i = 0;i < crfvot->num_features;++i) {
            const crfvol_feature_t* f = &crfvot->features[i];
            g[i] += (lbfgsi->sigma2inv * x[i]);
            norm += x[i] * x[i];
        }
        logl -= (lbfgsi->sigma2inv * norm * 0.5);
    }

    return -logl;
}

static int lbfgs_progress(
    void *instance,
    const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,
    const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step,
    int n,
    int k,
    int ls)
{
    int i, num_active_features = 0;
    clock_t duration, clk = clock();
    crfvol_t* crfvot = (crfvol_t*)instance;
    lbfgs_internal_t *lbfgsi = LBFGS_INTERNAL(crfvot);

    /* Compute the duration required for this iteration. */
    duration = clk - crfvot->clk_prev;
    crfvot->clk_prev = clk;

    /* Set feature weights from the L-BFGS solver. */
    for (i = 0;i < crfvot->num_features;++i) {
        lbfgsi->best_w[i] = x[i];
        if (x[i] != 0.) ++num_active_features;
    }

    /* Report the progress. */
    logging(crfvot->lg, "***** Iteration #%d *****\n", k);
    logging(crfvot->lg, "Log-likelihood: %f\n", -fx);
    logging(crfvot->lg, "Feature norm: %f\n", xnorm);
    logging(crfvot->lg, "Error norm: %f\n", gnorm);
    logging(crfvot->lg, "Active features: %d\n", num_active_features);
    logging(crfvot->lg, "Line search trials: %d\n", ls);
    logging(crfvot->lg, "Line search step: %f\n", step);
    logging(crfvot->lg, "Seconds required for this iteration: %.3f\n", duration / (double)CLOCKS_PER_SEC);

    /* Send the tagger with the current parameters. */
    if (crfvot->cbe_proc != NULL) {
        /* Callback notification with the tagger object. */
        int ret = crfvot->cbe_proc(crfvot->cbe_instance, &crfvot->tagger);
    }
    logging(crfvot->lg, "\n");

    /* Continue. */
    return 0;
}

int crfvol_lbfgs_options(crf_params_t* params, crfvol_option_t* opt, int mode)
{
    crfvol_lbfgs_option_t* lbfgs = &opt->lbfgs;

    BEGIN_PARAM_MAP(params, mode)
        DDX_PARAM_STRING(
            "regularization", lbfgs->regularization, "L2",
            "Specify the regularization type."
            )
        DDX_PARAM_FLOAT(
            "regularization.sigma", lbfgs->regularization_sigma, 10.0,
            "Specify the regularization constant."
            )
        DDX_PARAM_INT(
            "lbfgs.max_iterations", lbfgs->max_iterations, INT_MAX,
            "The maximum number of L-BFGS iterations."
            )
        DDX_PARAM_INT(
            "lbfgs.num_memories", lbfgs->memory, 6,
            "The number of corrections to approximate the inverse hessian matrix."
            )
        DDX_PARAM_FLOAT(
            "lbfgs.epsilon", lbfgs->epsilon, 1e-5,
            "Epsilon for testing the convergence of the objective."
            )
        DDX_PARAM_INT(
            "lbfgs.stop", lbfgs->stop, 10,
            "The duration of iterations to test the stopping criterion."
            )
        DDX_PARAM_FLOAT(
            "lbfgs.delta", lbfgs->delta, 1e-5,
            "The threshold for the stopping criterion; an L-BFGS iteration stops when the\n"
            "improvement of the log likelihood over the last ${lbfgs.stop} iterations is\n"
            "no greater than this threshold."
            )
        DDX_PARAM_STRING(
            "lbfgs.linesearch", lbfgs->linesearch, "MoreThuente",
            "The line search algorithm used in L-BFGS updates:\n"
            "{'MoreThuente': More and Thuente's method, 'Backtracking': backtracking}"
            )
        DDX_PARAM_INT(
            "lbfgs.linesearch.max_iterations", lbfgs->linesearch_max_iterations, 20,
            "The maximum number of trials for the line search algorithm."
            )
    END_PARAM_MAP()

    return 0;
}


int crfvol_lbfgs(
    crfvol_t* crfvot,
    crfvol_option_t *opt
    )
{
    int i, ret;
    const int K = crfvot->num_features;
    lbfgs_internal_t lbfgsi;
    lbfgs_parameter_t lbfgsparam;
    crfvol_lbfgs_option_t* lbfgsopt = &opt->lbfgs;

    /* Set the solver-specific information. */
    crfvot->solver_data = &lbfgsi;

    /* Allocate an array that stores the best weights. */
    lbfgsi.best_w = (floatval_t*)malloc(sizeof(floatval_t) * K);
    if (lbfgsi.best_w == NULL) {
        return CRFERR_OUTOFMEMORY;
    }
    for (i = 0;i < K;++i) {
        lbfgsi.best_w[i] = 0.;
    }

    /* Initialize the L-BFGS parameters with default values. */
    lbfgs_parameter_init(&lbfgsparam);

    logging(crfvot->lg, "L-BFGS optimization\n");
    logging(crfvot->lg, "regularization: %s\n", lbfgsopt->regularization);
    logging(crfvot->lg, "regularization.sigma: %f\n", lbfgsopt->regularization_sigma);
    logging(crfvot->lg, "lbfgs.num_memories: %d\n", lbfgsopt->memory);
    logging(crfvot->lg, "lbfgs.max_iterations: %d\n", lbfgsopt->max_iterations);
    logging(crfvot->lg, "lbfgs.epsilon: %f\n", lbfgsopt->epsilon);
    logging(crfvot->lg, "lbfgs.stop: %d\n", lbfgsopt->stop);
    logging(crfvot->lg, "lbfgs.delta: %f\n", lbfgsopt->delta);
    logging(crfvot->lg, "lbfgs.linesearch: %s\n", lbfgsopt->linesearch);
    logging(crfvot->lg, "lbfgs.linesearch.max_iterations: %d\n", lbfgsopt->linesearch_max_iterations);
    logging(crfvot->lg, "\n");

    /* Set parameters for L-BFGS. */
    lbfgsparam.m = lbfgsopt->memory;
    lbfgsparam.epsilon = lbfgsopt->epsilon;
    lbfgsparam.past = lbfgsopt->stop;
    lbfgsparam.delta = lbfgsopt->delta;
    lbfgsparam.max_iterations = lbfgsopt->max_iterations;
    if (strcmp(lbfgsopt->linesearch, "Backtracking") == 0) {
        lbfgsparam.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
    } else if (strcmp(lbfgsopt->linesearch, "StrongBacktracking") == 0) {
        lbfgsparam.linesearch = LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE;
    } else {
        lbfgsparam.linesearch = LBFGS_LINESEARCH_MORETHUENTE;
    }
    lbfgsparam.max_linesearch = lbfgsopt->linesearch_max_iterations;

    /* Set regularization parameters. */
    if (strcmp(lbfgsopt->regularization, "L1") == 0) {
        lbfgsi.l2_regularization = 0;
        lbfgsparam.orthantwise_c = 1.0 / lbfgsopt->regularization_sigma;
        lbfgsparam.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
    } else if (strcmp(lbfgsopt->regularization, "L2") == 0) {
        lbfgsi.l2_regularization = 1;
        lbfgsi.sigma2inv = 1.0 / (lbfgsopt->regularization_sigma * lbfgsopt->regularization_sigma);
        lbfgsparam.orthantwise_c = 0.;
    } else {
        lbfgsi.l2_regularization = 0;
        lbfgsparam.orthantwise_c = 0.;
    }

    /* Call the L-BFGS solver. */
    crfvot->clk_begin = clock();
    crfvot->clk_prev = crfvot->clk_begin;
    ret = lbfgs(
        crfvot->num_features,
        crfvot->w,
        NULL,
        lbfgs_evaluate,
        lbfgs_progress,
        crfvot,
        &lbfgsparam
        );
    if (ret == LBFGS_CONVERGENCE) {
        logging(crfvot->lg, "L-BFGS resulted in convergence\n");
    } else if (ret == LBFGS_STOP) {
        logging(crfvot->lg, "L-BFGS terminated with the stopping criteria\n");
    } else if (ret == LBFGSERR_MAXIMUMITERATION) {
        logging(crfvot->lg, "L-BFGS terminated with the maximum number of iterations\n");
    } else {
        logging(crfvot->lg, "L-BFGS terminated with error code (%d)\n", ret);
    }

    for (i = 0;i < K;++i) {
        crfvot->w[i] = lbfgsi.best_w[i];
    }

    logging(crfvot->lg, "Total seconds required for L-BFGS: %.3f\n", (clock() - crfvot->clk_begin) / (double)CLOCKS_PER_SEC);
    logging(crfvot->lg, "\n");

    return 0;
}
