/*
 * Use arbitrary precision matrix operations to compute a log likelihood.
 * The JSON format is used for both input and output.
 * Arbitrary precision is used only internally;
 * double precision floating point without error bounds
 * is used for input and output.
 *
 * The "probability_array" is a semantically flexible structure that defines
 * both the root prior distribution and the observations at the leaves.
 * Each site probability is the sum,
 * over all combinations of state assignments to nodes,
 * of the product of the state probabilities at nodes
 * times the product of transition probabilities on edges.
 *
 * For the log likelihood, the only available selection/aggregation
 * axis is the site axis. So the return value could consist of
 * an array of log likelihoods, or of summed, averaged, or linear
 * combinations of log likelihoods, optionally restricted to a
 * selection of sites.
 *
 * The output should be formatted in a way that is easily
 * readable as a data frame by the python module named pandas as follows:
 * >> import pandas as pd
 * >> f = open('output.json')
 * >> d = pd.read_json(f, orient='split', precise_float=True)
 *
 * On success, a json string is printed to stdout and the process terminates
 * with an exit status of zero.  On failure, information is written to stderr
 * and the process terminates with a nonzero exit status.
 *
 * input format:
 * {
 * "model_and_data" : {
 *  "edges" : [[a, b], [c, d], ...],                 (#edges, 2)
 *  "edge_rate_coefficients" : [a, b, ...],          (#edges, )
 *  "rate_matrix" : [[a, b, ...], [c, d, ...], ...], (#states, #states)
 *  "probability_array" : [...]                      (#sites, #nodes, #states)
 * },
 * "site_reduction" :
 * {
 *  "selection" : [a, b, c, ...], (optional)
 *  "aggregation" : {"sum" | "avg" | [a, b, c, ...]} (optional)
 * } (optional)
 * }
 *
 * output format (without aggregation of the "site" column):
 * {
 *  "columns" : ["site", "value"],
 *  "data" : [[a, b], [c, d], ..., [y, z]] (# selected sites)
 * }
 *
 * output format (with aggregation of the "site" column):
 * {
 *  "columns" : ["value"],
 *  "data" : [a]
 * }
 *
 */
#include "jansson.h"
#include "flint/flint.h"

#include "arb_mat.h"
#include "arb.h"
#include "arf.h"

#include "csr_graph.h"
#include "model.h"
#include "reduction.h"
#include "util.h"
#include "evaluate_site_lhood.h"
#include "equilibrium.h"
#include "arb_mat_extras.h"
#include "cross_site_ws.h"

#include "parsemodel.h"
#include "parsereduction.h"
#include "runjson.h"
#include "arbplfll.h"


/* This object changes in the course of the iteration over sites. */
typedef struct
{
    arb_mat_struct *base_node_vectors;
    arb_mat_struct *lhood_node_vectors;
} likelihood_ws_struct;
typedef likelihood_ws_struct likelihood_ws_t[1];

static void
likelihood_ws_init(likelihood_ws_t w, const model_and_data_t m)
{
    slong node_count = model_and_data_node_count(m);
    slong state_count = model_and_data_state_count(m);

    w->base_node_vectors = _arb_mat_vec_init(state_count, 1, node_count);
    w->lhood_node_vectors = _arb_mat_vec_init(state_count, 1, node_count);
}

static void
likelihood_ws_clear(likelihood_ws_t w, const model_and_data_t m)
{
    slong node_count = model_and_data_node_count(m);

    _arb_mat_vec_clear(w->base_node_vectors, node_count);
    _arb_mat_vec_clear(w->lhood_node_vectors, node_count);
}


static int
aggregate_across_sites(arb_t aggregate,
        const arb_struct *site_log_likelihoods,
        const column_reduction_t r, slong prec)
{
    slong i;
    int mode = r->agg_mode;

    arb_zero(aggregate);
    if (mode == AGG_SUM || mode == AGG_AVG)
    {
        for (i = 0; i < r->selection_len; i++)
        {
            arb_srcptr ll = site_log_likelihoods + r->selection[i];
            arb_add(aggregate, aggregate, ll, prec);
        }
        if (mode == AGG_AVG)
        {
            arb_div_si(aggregate, aggregate, r->selection_len, prec);
        }
    }
    else if (r->agg_mode == AGG_WEIGHTED_SUM)
    {
        arb_t weight;
        arb_init(weight);
        for (i = 0; i < r->selection_len; i++)
        {
            slong site = r->selection[i];
            arb_srcptr ll = site_log_likelihoods + site;
            arb_set_d(weight, r->weights[i]);
            arb_addmul(aggregate, ll, weight, prec);
        }
        arb_clear(weight);
    }
    else
    {
        fprintf(stderr, "error: unexpected aggregation mode\n");
        return -1;
    }

    return 0;
}


static json_t *
_query(model_and_data_t m, column_reduction_t r_site, int *result_out)
{
    json_t *j_out = NULL;
    int result = 0;
    int site_count = 0;
    int *site_is_selected = NULL;
    arb_t cat_lhood, prior_prob;
    arb_struct * site_likelihoods = NULL;
    arb_struct * site_log_likelihoods = NULL;
    arb_t aggregate;
    cross_site_ws_t csw;
    likelihood_ws_t w;
    int iter = 0;
    slong cat;
    slong ncats = 0;
    int i, site;
    slong prec = 4;
    int failed = 1;
    arb_ptr lhood, ll;

    arb_init(aggregate);
    arb_init(cat_lhood);
    arb_init(prior_prob);

    cross_site_ws_pre_init(csw);
    likelihood_ws_init(w, m);

    site_count = model_and_data_site_count(m);
    ncats = model_and_data_rate_category_count(m);

    site_is_selected = calloc(site_count, sizeof(int));
    for (i = 0; i < r_site->selection_len; i++)
    {
        site = r_site->selection[i];
        site_is_selected[site] = 1;
    }

    site_likelihoods = _arb_vec_init(site_count);
    site_log_likelihoods = _arb_vec_init(site_count);

    /* repeat site evaluations with increasing precision */
    while (failed)
    {
        failed = 0;
        cross_site_ws_reinit(csw, m, prec);

        /* if any likelihood is exactly zero then return an error */
        for (site = 0; site < site_count; site++)
        {
            lhood = site_likelihoods + site;
            ll = site_log_likelihoods + site;

            /* if the site is not in the selection then skip it */
            if (!site_is_selected[site])
                continue;

            /* if the log likelihood is fully evaluated then skip it */
            if (iter && r_site->agg_mode == AGG_NONE && _can_round(ll))
                continue;

            pmat_update_base_node_vectors(
                    w->base_node_vectors, m->p, site,
                    m->root_prior, csw->equilibrium,
                    m->preorder[0], prec);

            /* compute a weighted sum of per-category likelihoods */
            arb_zero(lhood);
            for (cat = 0; cat < ncats; cat++)
            {
                const arb_mat_struct *tmat_base;
                tmat_base = tmat_collection_entry(
                        csw->transition_matrices, cat, 0);
                evaluate_site_lhood(cat_lhood,
                        w->lhood_node_vectors,
                        NULL,
                        w->base_node_vectors,
                        tmat_base,
                        m->g, m->preorder, csw->node_count, prec);
                rate_mixture_get_prob(prior_prob, m->rate_mixture, cat, prec);
                arb_addmul(lhood, prior_prob, cat_lhood, prec);
            }

            if (arb_is_zero(lhood))
            {
                fprintf(stderr, "error: infeasible\n");
                result = -1;
                goto finish;
            }

            arb_log(ll, lhood, prec);

            /* if no aggregation and still bad bounds then fail */
            if (r_site->agg_mode == AGG_NONE && !_can_round(ll))
            {
                failed = 1;
            }
        }
        /* compute the aggregate if any */
        if (r_site->agg_mode != AGG_NONE)
        {
            result = aggregate_across_sites(
                    aggregate, site_log_likelihoods, r_site, prec);
            if (result)
            {
                goto finish;
            }
            if (!_can_round(aggregate))
            {
                failed = 1;
            }
        }
        prec <<= 1;
        iter++;
    }

    if (r_site->agg_mode == AGG_NONE)
    {
        double d;
        json_t *j_data, *x;
        j_data = json_array();
        for (i = 0; i < r_site->selection_len; i++)
        {
            site = r_site->selection[i];
            ll = site_log_likelihoods + site;
            d = arf_get_d(arb_midref(ll), ARF_RND_NEAR);
            x = json_pack("[i, f]", site, d);
            json_array_append_new(j_data, x);
        }
        j_out = json_pack("{s:[s, s], s:o}",
                "columns", "site", "value", "data", j_data);
    }
    else
    {
        j_out = json_pack("{s:[s], s:[f]}",
                "columns", "value", "data",
                arf_get_d(arb_midref(aggregate), ARF_RND_NEAR));
    }

finish:

    *result_out = result;

    arb_clear(aggregate);
    free(site_is_selected);
    arb_clear(cat_lhood);
    arb_clear(prior_prob);
    if (site_likelihoods)
    {
        _arb_vec_clear(site_likelihoods, site_count);
    }
    if (site_log_likelihoods)
    {
        _arb_vec_clear(site_log_likelihoods, site_count);
    }
    likelihood_ws_clear(w, m);
    cross_site_ws_clear(csw);
    return j_out;
}


static int
_parse(model_and_data_t m, column_reduction_t r_site, json_t *root)
{
    json_t *model_and_data = NULL;
    json_t *site_reduction = NULL;
    slong site_count;
    int result = 0;

    /* unpack the top level of json input */
    {
        size_t flags;
        json_error_t err;
        flags = JSON_STRICT;
        result = json_unpack_ex(root, &err, flags,
                "{s:o, s?o}",
                "model_and_data", &model_and_data,
                "site_reduction", &site_reduction
                );
        if (result)
        {
            fprintf(stderr, "error: on line %d: %s\n", err.line, err.text);
            return result;
        }
    }

    /* validate the model and data section of the json input */
    result = validate_model_and_data(m, model_and_data);
    if (result) return result;

    /* initialize counts */
    site_count = model_and_data_site_count(m);

    /* validate the site reduction section of the json input */
    result = validate_column_reduction(
            r_site, site_count, "site", site_reduction);
    if (result) return result;

    return result;
}


json_t *arbplf_ll_run(void *userdata, json_t *root, int *retcode)
{
    json_t *j_out = NULL;
    model_and_data_t m;
    column_reduction_t r_site;
    int result = 0;

    model_and_data_init(m);
    column_reduction_init(r_site);

    if (userdata)
    {
        fprintf(stderr, "internal error: unexpected userdata\n");
        result = -1;
        goto finish;
    }

    result = _parse(m, r_site, root);
    if (result) goto finish;

    j_out = _query(m, r_site, &result);
    if (result) goto finish;

finish:

    *retcode = result;

    model_and_data_clear(m);
    column_reduction_clear(r_site);

    flint_cleanup();
    return j_out;
}
