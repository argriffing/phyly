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

#include "parsemodel.h"
#include "parsereduction.h"
#include "runjson.h"
#include "arbplfll.h"


/*
 * Likelihood workspace.
 * The object lifetime is limited to only one level of precision,
 * but it extends across site evaluations.
 */
typedef struct
{
    slong prec;
    int node_count;
    int edge_count;
    int state_count;
    arb_struct *edge_rates;
    arb_mat_t rate_matrix;
    arb_mat_struct *transition_matrices;
    arb_mat_struct *base_node_column_vectors;
    arb_mat_struct *lhood_node_column_vectors;
} likelihood_ws_struct;
typedef likelihood_ws_struct likelihood_ws_t[1];

static void
likelihood_ws_init(likelihood_ws_t w, model_and_data_t m, slong prec)
{
    csr_graph_struct *g;
    int i, j, k;
    arb_mat_struct * tmat;
    arb_mat_struct * nmat;
    double tmpd;

    if (!m)
    {
        arb_mat_init(w->rate_matrix, 0, 0);
        w->transition_matrices = NULL;
        w->base_node_column_vectors = NULL;
        w->lhood_node_column_vectors = NULL;
        w->edge_rates = NULL;
        w->node_count = 0;
        w->edge_count = 0;
        w->state_count = 0;
        w->prec = 0;
        return;
    }

    g = m->g;

    w->prec = prec;
    w->node_count = g->n;
    w->edge_count = g->nnz;
    w->state_count = arb_mat_nrows(m->mat);

    w->edge_rates = _arb_vec_init(w->edge_count);
    arb_mat_init(w->rate_matrix, w->state_count, w->state_count);
    w->transition_matrices = flint_malloc(
            w->edge_count * sizeof(arb_mat_struct));
    w->base_node_column_vectors = flint_malloc(
            w->node_count * sizeof(arb_mat_struct));
    w->lhood_node_column_vectors = flint_malloc(
            w->node_count * sizeof(arb_mat_struct));

    /*
     * This is the csr graph index of edge (a, b).
     * Given this index, node b is directly available
     * from the csr data structure.
     * The rate coefficient associated with the edge will also be available.
     * On the other hand, the index of node 'a' will be available through
     * the iteration order rather than directly from the index.
     */
    int idx;

    /*
     * Define the map from csr edge index to edge rate.
     * The edge rate is represented in arbitrary precision.
     */
    if (!m->edge_map)
    {
        fprintf(stderr, "internal error: edge map is uninitialized\n");
        abort();
    }
    if (!m->edge_map->order)
    {
        fprintf(stderr, "internal error: edge map order is uninitialized\n");
        abort();
    }
    if (!m->edge_rate_coefficients)
    {
        fprintf(stderr, "internal error: edge rate coeffs unavailable\n");
        abort();
    }
    for (i = 0; i < w->edge_count; i++)
    {
        idx = m->edge_map->order[i];
        tmpd = m->edge_rate_coefficients[i];
        arb_set_d(w->edge_rates + idx, tmpd);
    }

    /* Initialize the unscaled arbitrary precision rate matrix. */
    dmat_get_arb_mat(w->rate_matrix, m->mat);
    _arb_mat_scalar_div_d(w->rate_matrix, m->rate_divisor, prec);

    /*
     * Modify the diagonals of the unscaled rate matrix
     * so that the sum of each row is zero.
     */
    _arb_update_rate_matrix_diagonal(w->rate_matrix, w->prec);

    /*
     * Initialize the array of arbitrary precision transition matrices.
     * They will initially contain appropriately scaled rate matrices.
     * Although the unscaled rates have zero arb radius and the
     * scaling coefficients have zero arb radius, the entries of the
     * scaled rate matrices will in general have positive arb radius.
     */
    for (idx = 0; idx < w->edge_count; idx++)
    {
        tmat = w->transition_matrices + idx;
        arb_mat_init(tmat, w->state_count, w->state_count);
    }
    for (idx = 0; idx < w->edge_count; idx++)
    {
        tmat = w->transition_matrices + idx;
        for (j = 0; j < w->state_count; j++)
        {
            for (k = 0; k < w->state_count; k++)
            {
                arb_mul(arb_mat_entry(tmat, j, k),
                        arb_mat_entry(w->rate_matrix, j, k),
                        w->edge_rates + idx, w->prec);
            }
        }
    }

    /*
     * Compute the matrix exponentials of the scaled transition rate matrices.
     * Note that the arb matrix exponential function allows aliasing,
     * so we do not need to allocate a temporary array (although a temporary
     * array will be created by the arb function).
     */
    for (idx = 0; idx < w->edge_count; idx++)
    {
        tmat = w->transition_matrices + idx;
        arb_mat_exp(tmat, tmat, w->prec);
    }

    /*
     * Allocate an arbitrary precision column vector for each node.
     * These are for base likelihoods involving data or the prior.
     */
    for (i = 0; i < w->node_count; i++)
    {
        nmat = w->base_node_column_vectors + i;
        arb_mat_init(nmat, w->state_count, 1);
    }

    /*
     * Allocate an arbitrary precision column vector for each node.
     * This will be used to accumulate conditional likelihoods.
     */
    for (i = 0; i < w->node_count; i++)
    {
        nmat = w->lhood_node_column_vectors + i;
        arb_mat_init(nmat, w->state_count, 1);
    }
}

static void
likelihood_ws_clear(likelihood_ws_t w)
{
    int i, idx;

    if (w->edge_rates)
    {
        _arb_vec_clear(w->edge_rates, w->edge_count);
    }
    arb_mat_clear(w->rate_matrix);

    for (idx = 0; idx < w->edge_count; idx++)
    {
        arb_mat_clear(w->transition_matrices + idx);
    }
    flint_free(w->transition_matrices);
    
    for (i = 0; i < w->node_count; i++)
    {
        arb_mat_clear(w->base_node_column_vectors + i);
    }
    flint_free(w->base_node_column_vectors);

    for (i = 0; i < w->node_count; i++)
    {
        arb_mat_clear(w->lhood_node_column_vectors + i);
    }
    flint_free(w->lhood_node_column_vectors);
}


json_t *arbplf_ll_run(void *userdata, json_t *root, int *retcode)
{
    json_t *j_out = NULL;
    json_t *model_and_data = NULL;
    json_t *site_reduction = NULL;
    int result = 0;
    int site_count = 0;
    model_and_data_t m;
    column_reduction_t r;
    int *site_is_selected = NULL;
    arb_struct * site_likelihoods = NULL;
    arb_struct * site_log_likelihoods = NULL;
    arb_t aggregate, tmp, weight;
    likelihood_ws_t w;
    int iter = 0;

    arb_init(tmp);
    arb_init(weight);
    arb_init(aggregate);

    likelihood_ws_init(w, NULL, 0);
    model_and_data_init(m);
    column_reduction_init(r);

    if (userdata)
    {
        fprintf(stderr, "error: unexpected userdata\n");
        result = -1;
        goto finish;
    }

    /* parse the json input */
    {
        json_error_t err;
        size_t flags;
        
        flags = JSON_STRICT;
        result = json_unpack_ex(root, &err, flags,
                "{s:o, s?o}",
                "model_and_data", &model_and_data,
                "site_reduction", &site_reduction
                );
        if (result)
        {
            fprintf(stderr, "error: on line %d: %s\n", err.line, err.text);
            goto finish;
        }
    }

    /* validate the model and data section of the json input */
    result = validate_model_and_data(m, model_and_data);
    if (result) goto finish;

    site_count = pmat_nsites(m->p);

    /* validate the (optional) site reduction section of the json input */
    result = validate_column_reduction(r, site_count, "site", site_reduction);
    if (result) goto finish;

    int i;
    int site;
    site_is_selected = calloc(site_count, sizeof(int));
    for (i = 0; i < r->selection_len; i++)
    {
        site = r->selection[i];
        site_is_selected[site] = 1;
    }

    site_likelihoods = _arb_vec_init(site_count);
    site_log_likelihoods = _arb_vec_init(site_count);

    /* repeat site evaluations with increasing precision */
    slong prec = 4;
    int failed = 1;
    arb_ptr lhood, ll;
    while (failed)
    {
        failed = 0;
        likelihood_ws_clear(w);
        likelihood_ws_init(w, m, prec);
        /* if any likelihood is exactly zero then return an error */
        for (site = 0; site < site_count; site++)
        {
            lhood = site_likelihoods + site;
            ll = site_log_likelihoods + site;

            /* if the site is not in the selection then skip it */
            if (!site_is_selected[site])
                continue;

            /* if the log likelihood is fully evaluated then skip it */
            if (iter && r->agg_mode == AGG_NONE && _can_round(ll))
                continue;

            pmat_update_base_node_vectors(
                    w->base_node_column_vectors, m->p, site);

            evaluate_site_lhood(lhood,
                    w->lhood_node_column_vectors,
                    NULL,
                    w->base_node_column_vectors,
                    w->transition_matrices,
                    m->g, m->preorder, w->node_count, w->prec);

            if (arb_is_zero(lhood))
            {
                fprintf(stderr, "error: infeasible\n");
                result = -1;
                goto finish;
            }
            arb_log(ll, lhood, prec);
            /* if no aggregation and still bad bounds then fail */
            if (r->agg_mode == AGG_NONE && !_can_round(ll))
            {
                failed = 1;
            }
        }
        /* compute the aggregate if any */
        if (r->agg_mode != AGG_NONE)
        {
            arb_zero(aggregate);
            arb_one(weight);
            for (i = 0; i < r->selection_len; i++)
            {
                site = r->selection[i];
                if (r->agg_mode == AGG_WEIGHTED_SUM)
                {
                    arb_set_d(weight, r->weights[i]);
                }
                ll = site_log_likelihoods + site;
                arb_addmul(aggregate, ll, weight, prec);
            }
            if (r->agg_mode == AGG_AVG)
            {
                arb_div_si(aggregate, aggregate, r->selection_len, prec);
            }
            /* if aggregate has bad bounds then fail */
            if (!_can_round(aggregate))
            {
                failed = 1;
            }
        }
        prec <<= 1;
        iter++;
    }

    if (r->agg_mode == AGG_NONE)
    {
        double d;
        json_t *j_data, *x;
        j_data = json_array();
        for (i = 0; i < r->selection_len; i++)
        {
            site = r->selection[i];
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

    *retcode = result;

    arb_clear(aggregate);
    arb_clear(tmp);
    arb_clear(weight);
    free(site_is_selected);
    if (site_likelihoods)
    {
        _arb_vec_clear(site_likelihoods, site_count);
    }
    if (site_log_likelihoods)
    {
        _arb_vec_clear(site_log_likelihoods, site_count);
    }
    column_reduction_clear(r);
    model_and_data_clear(m);
    likelihood_ws_clear(w);
    flint_cleanup();
    return j_out;
}
