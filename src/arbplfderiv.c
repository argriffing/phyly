/*
 * Use arbitrary precision matrix operations to compute phylogenetic
 * log likelihood derivatives with respect to edge rate coefficients.
 *
 * The JSON format is used for both input and output.
 * Arbitrary precision is used only internally;
 * double precision floating point without error bounds
 * is used for input and output.
 *
 * See the arbplfll.c comments (or even the docs if they have
 * been written by now...) for more details.
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
 * }, (optional)
 * "edge_reduction" :
 * {
 *  "selection" : [a, b, c, ...], (optional)
 *  "aggregation" : {"sum" | "avg" | [a, b, c, ...]} (optional)
 * } (optional)
 * }
 *
 * output format (with no aggregation):
 * {
 *  "columns" : ["site", "edge", "value"],
 *  "data" : [[a, b, c], [d, e, f], ..., [x, y, z]]
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
#include "arbplfderiv.h"

typedef struct
{
    arb_mat_struct *base_node_vectors;
    arb_mat_struct *lhood_node_vectors;
    arb_mat_struct *lhood_edge_vectors;
    arb_mat_struct *deriv_node_vectors;
} likelihood_ws_struct;
typedef likelihood_ws_struct likelihood_ws_t[1];

static void
likelihood_ws_init(likelihood_ws_t w, const model_and_data_t m)
{
    slong edge_count = model_and_data_edge_count(m);
    slong node_count = model_and_data_node_count(m);
    slong state_count = model_and_data_state_count(m);

    w->lhood_edge_vectors = _arb_mat_vec_init(state_count, 1, edge_count);
    w->base_node_vectors = _arb_mat_vec_init(state_count, 1, node_count);
    w->lhood_node_vectors = _arb_mat_vec_init(state_count, 1, node_count);
    w->deriv_node_vectors = _arb_mat_vec_init(state_count, 1, node_count);
}

static void
likelihood_ws_clear(likelihood_ws_t w, const model_and_data_t m)
{
    slong edge_count = model_and_data_edge_count(m);
    slong node_count = model_and_data_node_count(m);

    _arb_mat_vec_clear(w->lhood_edge_vectors, edge_count);
    _arb_mat_vec_clear(w->base_node_vectors, node_count);
    _arb_mat_vec_clear(w->lhood_node_vectors, node_count);
    _arb_mat_vec_clear(w->deriv_node_vectors, node_count);
}


/* Helper function to update likelihood-related vectors at a site. */
static int
_update_lhood_vectors(arb_t lhood,
        model_and_data_t m, cross_site_ws_t csw, likelihood_ws_t w,
        slong site, slong prec)
{
    arb_mat_struct *tmat_base;
    slong cat;
    slong node_count = model_and_data_node_count(m);

    /* todo: allow more than one category */
    cat = 0;

    tmat_base = tmat_collection_entry(csw->transition_matrices, cat, 0);

    pmat_update_base_node_vectors(
            w->base_node_vectors, m->p, site,
            m->root_prior, csw->equilibrium,
            m->preorder[0], prec);

    evaluate_site_lhood(lhood,
            w->lhood_node_vectors,
            w->lhood_edge_vectors,
            w->base_node_vectors,
            tmat_base,
            m->g, m->preorder, node_count, prec);

    /* todo: allow individual likelihoods in the mixture to be zero */
    if (arb_is_zero(lhood))
    {
        fprintf(stderr, "error: infeasible\n");
        return -1;
    }

    return 0;
}


/*
 * Evaluate derivatives of likelihood with respect to edge rate coefficients.
 *
 * NOTE !!! this function requires the site likelihood to have been
 * computed for this site, and requires the associated intermediate
 * calculations on edges and nodes to be available in the workspace.
 *
 * Only evaluate derivatives for the selected edges.
 * Within this function, csr graph indices are used to identify edges.
 * So for example, these are used for indexing the derivatives array
 * and for indexing the edge_selection_count array.
 *
 * The idx_to_a and b_to_idx arrays define the maps
 * edge->initial_node and final_node->edge respectively.
 */
static void
evaluate_site_derivatives(arb_struct *derivatives, int *edge_is_requested,
        model_and_data_t m, cross_site_ws_t csw, likelihood_ws_t w,
        int *idx_to_a, int *b_to_idx, int site, slong prec)
{
    int a, b, idx;
    int start, stop;
    int state;
    double tmpd;
    arb_mat_struct * nmat;
    arb_mat_struct * nmatb;
    arb_mat_struct * tmat;
    arb_mat_struct * emat;
    arb_mat_struct * rmat;
    csr_graph_struct *g;
    arb_mat_struct *tmat_base;
    slong cat;

    slong state_count = model_and_data_state_count(m);
    slong edge_count = model_and_data_edge_count(m);

    /* todo: allow more than one rate category */
    cat = 0;
    tmat_base = tmat_collection_entry(csw->transition_matrices, cat, 0);

    g = m->g;
    _arb_vec_zero(derivatives, edge_count);

    /*
     * For each requested edge at this site,
     * compute the state vectors associated with the derivative.
     * This involves tracing back from the edge to the root.
     */
    int deriv_idx;
    int curr_idx;
    for (deriv_idx = 0; deriv_idx < edge_count; deriv_idx++)
    {
        if (!edge_is_requested[deriv_idx])
            continue;

        curr_idx = deriv_idx;
        while (curr_idx != -1)
        {
            a = idx_to_a[curr_idx];

            /* todo: could this step simply copy the base node vector
             * instead of duplicating the base node vector creation code?
             */
            /* initialize the state vector for node a */
            nmat = w->deriv_node_vectors + a;
            for (state = 0; state < state_count; state++)
            {
                tmpd = *pmat_srcentry(m->p, site, a, state);
                arb_set_d(arb_mat_entry(nmat, state, 0), tmpd);
            }
            if (a == m->preorder[0])
            {
                root_prior_mul_col_vec(
                        nmat, m->root_prior, csw->equilibrium, prec);
            }

            /*
             * Multiplicatively accumulate precomputed vectors
             * for each edge except for the 'current' edge that is on the
             * path from the root to the edge of interest for the derivative.
             * The accumulation associated with the current edge
             * will involve a matrix-vector product.
             */
            start = g->indptr[a];
            stop = g->indptr[a+1];
            for (idx = start; idx < stop; idx++)
            {
                if (idx == deriv_idx)
                {
                    rmat = csw->rate_matrix;
                    emat = w->lhood_edge_vectors + idx;
                    _prune_update(nmat, nmat, rmat, emat, prec);
                }
                else if (idx == curr_idx)
                {
                    b = g->indices[idx];
                    tmat = tmat_base + idx;
                    nmatb = w->deriv_node_vectors + b;
                    _prune_update(nmat, nmat, tmat, nmatb, prec);
                }
                else
                {
                    emat = w->lhood_edge_vectors + idx;
                    _arb_mat_mul_entrywise(nmat, nmat, emat, prec);
                }
            }

            /* move back towards the root of the tree */
            curr_idx = b_to_idx[a];
        }

        /* Report the sum of state entries associated with the root. */
        nmat = w->deriv_node_vectors + m->root_node_index;
        arb_struct * deriv = derivatives + deriv_idx;
        arb_zero(deriv);
        for (state = 0; state < state_count; state++)
        {
            arb_add(deriv, deriv, arb_mat_entry(nmat, state, 0), prec);
        }
    }

}






static json_t *
_agg_site_edge_yy(
        model_and_data_t m,
        int *idx_to_a, int *b_to_idx,
        column_reduction_t r_site, column_reduction_t r_edge,
        int *site_selection_count, int *edge_selection_count,
        int *result_out)
{
    json_t * j_out = NULL;
    cross_site_ws_t csw;
    likelihood_ws_t w;
    int result = 0;
    int site;
    slong prec;
    int failed;

    int site_count = pmat_nsites(m->p);
    int edge_count = m->g->nnz;

    arb_t site_likelihood;
    arb_t deriv_lhood_edge_accum;
    arb_t deriv_ll_site_accum;
    arb_struct * derivatives;

    arb_init(site_likelihood);
    arb_init(deriv_lhood_edge_accum);
    arb_init(deriv_ll_site_accum);
    derivatives = _arb_vec_init(edge_count);

    arb_struct * edge_weights;
    arb_t edge_weight_divisor;

    arb_init(edge_weight_divisor);
    edge_weights = _arb_vec_init(edge_count);

    arb_struct * site_weights;
    arb_t site_weight_divisor;

    arb_init(site_weight_divisor);
    site_weights = _arb_vec_init(site_count);

    cross_site_ws_pre_init(csw);
    likelihood_ws_init(w, m);

    /*
     * The recipe is:
     * 1) repeat with increasing precision until arb gives the green light:
     *    2) for each site:
     *       2a) aggregate derivatives of likelihood across edges
     *       2b) divide the aggregate by the likelihood, thereby
     *           converting from derivative of likelihood to derivative
     *           of log likelihood
     *    3) aggregate those aggregates across sites
     *
     * Note that this uses the linearity of the aggregate functions
     * in a nontrivial way in step (2b).
     */

    /* repeat with increasing precision until there is no precision failure */
    for (failed=1, prec=4; failed; prec<<=1)
    {
        /* define edge aggregation weights with respect to csr indices */
        result = get_edge_agg_weights(
                edge_weight_divisor, edge_weights,
                edge_count, m->edge_map->order, r_edge, prec);
        if (result) goto finish;

        /* define site aggregation weights */
        result = get_site_agg_weights(
                site_weight_divisor, site_weights, site_count, r_site, prec);
        if (result) goto finish;

        cross_site_ws_reinit(csw, m, prec);

        /* clear the accumulation across sites */
        arb_zero(deriv_ll_site_accum);

        for (site = 0; site < site_count; site++)
        {
            if (!site_selection_count[site])
                continue;

            result = _update_lhood_vectors(
                    site_likelihood, m, csw, w, site, prec);
            if (result) goto finish;

            /*
             * Evaluate derivatives of site likelihood
             * with respect to each selected edge.
             * Note that these derivatives are with respect to the likelihood
             * rather than with respect to the log likelihood.
             */
            evaluate_site_derivatives(
                    derivatives,
                    edge_selection_count,
                    m, csw, w,
                    idx_to_a, b_to_idx, site, prec);

            /*
             * Aggregate the site derivatives.
             * Divide the aggregate by the site likelihood.
             * The accumulation across edges does not need to be cleared,
             * because it is set directly by computing a dot product.
             */
            _arb_vec_dot(deriv_lhood_edge_accum,
                    derivatives, edge_weights, edge_count, prec);

            arb_div(deriv_lhood_edge_accum,
                    deriv_lhood_edge_accum, edge_weight_divisor, prec);

            arb_div(deriv_lhood_edge_accum,
                    deriv_lhood_edge_accum, site_likelihood, prec);

            /* Add to the across-site aggregate. */
            arb_addmul(deriv_ll_site_accum,
                    deriv_lhood_edge_accum, site_weights+site, prec);
        }

        /*
         * Finish aggregating across sites.
         * Check the bounds of the solution.
         */
        arb_div(deriv_ll_site_accum,
                deriv_ll_site_accum, site_weight_divisor, prec);

        failed = !_can_round(deriv_ll_site_accum);
    }

    if (failed)
    {
        fprintf(stderr, "internal error: insufficient precision\n");
        result = -1;
        goto finish;
    }

    j_out = json_pack("{s:[s], s:[f]}",
            "columns", "value", "data",
            arf_get_d(arb_midref(deriv_ll_site_accum), ARF_RND_NEAR));

finish:

    likelihood_ws_clear(w, m);
    arb_clear(site_likelihood);
    arb_clear(deriv_lhood_edge_accum);
    arb_clear(deriv_ll_site_accum);
    _arb_vec_clear(derivatives, edge_count);

    arb_clear(site_weight_divisor);
    _arb_vec_clear(site_weights, site_count);

    arb_clear(edge_weight_divisor);
    _arb_vec_clear(edge_weights, edge_count);

    *result_out = result;
    return j_out;
}


static json_t *
_agg_site_edge_yn(
        model_and_data_t m,
        int *idx_to_a, int *b_to_idx,
        column_reduction_t r_site, column_reduction_t r_edge,
        int *site_selection_count, int *edge_selection_count,
        int *result_out)
{
    json_t * j_out = NULL;
    cross_site_ws_t csw;
    likelihood_ws_t w;
    int result = 0;
    int i, idx, site, edge;
    slong prec;
    int failed;

    int *edge_is_requested = NULL;

    int site_count = pmat_nsites(m->p);
    int edge_count = m->g->nnz;

    arb_t x;
    arb_t site_likelihood;
    arb_struct * final;
    arb_struct * derivatives;
    arb_struct * deriv_ll_accum;

    arb_init(x);
    arb_init(site_likelihood);
    derivatives = _arb_vec_init(edge_count);
    deriv_ll_accum = _arb_vec_init(edge_count);
    final = _arb_vec_init(edge_count);

    arb_struct * site_weights;
    arb_t site_weight_divisor;

    arb_init(site_weight_divisor);
    site_weights = _arb_vec_init(site_count);

    cross_site_ws_pre_init(csw);
    likelihood_ws_init(w, m);

    /*
     * Initially we want edge derivatives that have been selected.
     * Later we may stop requesting derivatives for edges
     * for which the derivative (aggregated across sites) has been
     * determined to have sufficient accuracy.
     */
    edge_is_requested = malloc(edge_count * sizeof(int));
    for (idx = 0; idx < edge_count; idx++)
    {
        edge_is_requested[idx] = edge_selection_count[idx] != 0;
    }

    /* repeat with increasing precision until there is no precision failure */
    for (failed=1, prec=4; failed; prec<<=1)
    {
        /* define site aggregation weights */
        result = get_site_agg_weights(
                site_weight_divisor, site_weights, site_count, r_site, prec);
        if (result) goto finish;

        cross_site_ws_reinit(csw, m, prec);

        /* clear the accumulation across sites */
        _arb_vec_zero(deriv_ll_accum, edge_count);

        for (site = 0; site < site_count; site++)
        {
            if (!site_selection_count[site])
                continue;

            result = _update_lhood_vectors(
                    site_likelihood, m, csw, w, site, prec);
            if (result) goto finish;

            /*
             * Evaluate derivatives of site likelihood
             * with respect to each selected edge.
             * Note that these derivatives are with respect to the likelihood
             * rather than with respect to the log likelihood.
             */
            evaluate_site_derivatives(
                    derivatives,
                    edge_is_requested,
                    m, csw, w,
                    idx_to_a, b_to_idx, site, prec);

            /* define a site-specific coefficient */
            arb_div(x, site_weights+site, site_weight_divisor, prec);
            arb_div(x, x, site_likelihood, prec);

            /* accumulate */
            for (idx = 0; idx < edge_count; idx++)
            {
                if (edge_is_requested[idx])
                {
                    /*
                    flint_printf("debug edge deriv info:\n");
                    arb_printd(x, 15); flint_printf("\n");
                    arb_printd(derivatives + idx, 15); flint_printf("\n");
                    flint_printf("\n");
                    */
                    arb_addmul(deriv_ll_accum + idx, derivatives + idx, x, prec);
                }
            }
        }

        /* check bounds */
        failed = 0;
        for (idx = 0; idx < edge_count; idx++)
        {
            if (edge_is_requested[idx])
            {
                if (_can_round(deriv_ll_accum + idx))
                {
                    /*
                    printf("debug: can round idx=%d prec=%d\n",
                            idx, (int) prec);
                    */
                    edge_is_requested[idx] = 0;
                    arb_set(final + idx, deriv_ll_accum + idx);
                }
                else
                {
                    /*
                    printf("debug: cannot round idx=%d prec=%d\n",
                            idx, (int) prec);
                    arb_printd(deriv_ll_accum + idx, 15); flint_printf("\n");
                    */
                    failed = 1;
                }
            }
        }
    }

    if (failed)
    {
        fprintf(stderr, "internal error: insufficient precision\n");
        result = -1;
        goto finish;
    }

    /* build the json output */
    {
        double d;
        json_t *j_data, *x;
        j_data = json_array();
        for (i = 0; i < r_edge->selection_len; i++)
        {
            edge = r_edge->selection[i];
            idx = m->edge_map->order[edge];
            d = arf_get_d(arb_midref(final + idx), ARF_RND_NEAR);
            x = json_pack("[i, f]", edge, d);
            json_array_append_new(j_data, x);
        }
        j_out = json_pack("{s:[s, s], s:o}",
                "columns", "edge", "value",
                "data", j_data);
    }

finish:

    free(edge_is_requested);
    likelihood_ws_clear(w, m);

    arb_clear(x);
    arb_clear(site_likelihood);
    _arb_vec_clear(derivatives, edge_count);
    _arb_vec_clear(deriv_ll_accum, edge_count);
    _arb_vec_clear(final, edge_count);

    arb_clear(site_weight_divisor);
    _arb_vec_clear(site_weights, site_count);

    *result_out = result;
    return j_out;
}


static json_t *
_agg_site_edge_ny(
        model_and_data_t m,
        int *idx_to_a, int *b_to_idx,
        column_reduction_t r_site, column_reduction_t r_edge,
        int *site_selection_count, int *edge_selection_count,
        int *result_out)
{
    json_t * j_out = NULL;
    cross_site_ws_t csw;
    likelihood_ws_t w;
    int result = 0;
    int i, site;
    slong prec;
    int failed;

    int *site_is_requested = NULL;

    int site_count = pmat_nsites(m->p);
    int edge_count = m->g->nnz;

    arb_struct * p;
    arb_t site_likelihood;
    arb_struct * final;
    arb_struct * derivatives;

    arb_init(site_likelihood);
    derivatives = _arb_vec_init(edge_count);
    final = _arb_vec_init(site_count);

    arb_struct * edge_weights;
    arb_t edge_weight_divisor;

    arb_init(edge_weight_divisor);
    edge_weights = _arb_vec_init(edge_count);

    cross_site_ws_pre_init(csw);
    likelihood_ws_init(w, m);

    /*
     * Initially we want edge derivatives that have been selected.
     * Later we may stop requesting derivatives for edges
     * for which the derivative (aggregated across sites) has been
     * determined to have sufficient accuracy.
     */
    site_is_requested = malloc(site_count * sizeof(int));
    for (site = 0; site < site_count; site++)
    {
        site_is_requested[site] = site_selection_count[site] != 0;
    }

    /* repeat with increasing precision until there is no precision failure */
    for (failed=1, prec=4; failed; prec<<=1)
    {
        /* define edge aggregation weights with respect to csr indices */
        result = get_edge_agg_weights(
                edge_weight_divisor, edge_weights,
                edge_count, m->edge_map->order, r_edge, prec);
        if (result) goto finish;

        cross_site_ws_reinit(csw, m, prec);

        /* update edge derivative accumulations at requested sites */
        for (site = 0; site < site_count; site++)
        {
            if (!site_is_requested[site])
                continue;

            result = _update_lhood_vectors(
                    site_likelihood, m, csw, w, site, prec);
            if (result) goto finish;

            evaluate_site_derivatives(
                    derivatives,
                    edge_selection_count,
                    m, csw, w,
                    idx_to_a, b_to_idx, site, prec);
            p = final + site;
            _arb_vec_dot(p, derivatives, edge_weights, edge_count, prec);
            arb_div(p, p, edge_weight_divisor, prec);
            arb_div(p, p, site_likelihood, prec);
        }

        /* check bounds */
        failed = 0;
        for (site = 0; site < site_count; site++)
        {
            if (site_is_requested[site])
            {
                if (_can_round(final + site))
                {
                    site_is_requested[site] = 0;
                }
                else
                {
                    failed = 1;
                }
            }
        }
    }

    if (failed)
    {
        fprintf(stderr, "internal error: insufficient precision\n");
        result = -1;
        goto finish;
    }

    /* build the json output */
    {
        double d;
        json_t *j_data, *x;
        j_data = json_array();
        for (i = 0; i < r_site->selection_len; i++)
        {
            site = r_site->selection[i];
            d = arf_get_d(arb_midref(final + site), ARF_RND_NEAR);
            x = json_pack("[i, f]", site, d);
            json_array_append_new(j_data, x);
        }
        j_out = json_pack("{s:[s, s], s:o}",
                "columns", "site", "value",
                "data", j_data);
    }

finish:

    free(site_is_requested);
    likelihood_ws_clear(w, m);

    arb_clear(site_likelihood);
    _arb_vec_clear(derivatives, edge_count);
    _arb_vec_clear(final, site_count);

    arb_clear(edge_weight_divisor);
    _arb_vec_clear(edge_weights, edge_count);

    *result_out = result;
    return j_out;
}


static json_t *
_agg_site_edge_nn(
        model_and_data_t m,
        int *idx_to_a, int *b_to_idx,
        column_reduction_t r_site, column_reduction_t r_edge,
        int *site_selection_count, int *edge_selection_count,
        int *result_out)
{
    json_t * j_out = NULL;
    cross_site_ws_t csw;
    likelihood_ws_t w;
    int result = 0;
    int i, idx, edge, site;
    slong prec;
    int failed;

    int *site_is_requested = NULL;
    int *edge_is_requested = NULL;

    int site_count = pmat_nsites(m->p);
    int edge_count = m->g->nnz;

    arb_struct * p;
    arb_t site_likelihood;
    arb_mat_t final;
    arb_struct * derivatives;

    arb_init(site_likelihood);
    derivatives = _arb_vec_init(edge_count);
    arb_mat_init(final, site_count, edge_count);

    cross_site_ws_pre_init(csw);
    likelihood_ws_init(w, m);

    /* These will be updated each time the precision is bumped. */
    site_is_requested = malloc(site_count * sizeof(int));
    for (site = 0; site < site_count; site++)
    {
        site_is_requested[site] = site_selection_count[site] != 0;
    }
    edge_is_requested = malloc(edge_count * sizeof(int));
    for (idx = 0; idx < edge_count; idx++)
    {
        edge_is_requested[idx] = edge_selection_count[idx] != 0;
    }

    /* repeat with increasing precision until there is no precision failure */
    for (failed=1, prec=4; failed; prec<<=1)
    {
        cross_site_ws_reinit(csw, m, prec);

        /* update requested sites */
        for (site = 0; site < site_count; site++)
        {
            if (!site_is_requested[site])
                continue;

            result = _update_lhood_vectors(
                    site_likelihood, m, csw, w, site, prec);
            if (result) goto finish;

            evaluate_site_derivatives(
                    derivatives,
                    edge_is_requested,
                    m, csw, w,
                    idx_to_a, b_to_idx, site, prec);
            for (idx = 0; idx < edge_count; idx++)
            {
                if (edge_is_requested[idx])
                {
                    arb_div(arb_mat_entry(final, site, idx),
                            derivatives + idx, site_likelihood, prec);
                }
            }
        }

        /* check bounds */
        {
            failed = 0;
            int req_site;
            int req_edge;
            for (site = 0; site < site_count; site++) {
                if (site_is_requested[site]) {
                    req_site = 0;
                    for (idx = 0; idx < edge_count; idx++) {
                        if (edge_is_requested[idx]) {
                            p = arb_mat_entry(final, site, idx);
                            if (!_can_round(p)) {
                                req_site = 1;
                                failed = 1;
                            }
                        }
                    }
                    site_is_requested[site] = req_site;
                }
            }
            for (idx = 0; idx < edge_count; idx++) {
                if (edge_is_requested[idx]) {
                    req_edge = 0;
                    for (site = 0; site < site_count; site++) {
                        if (site_is_requested[site]) {
                            p = arb_mat_entry(final, site, idx);
                            if (!_can_round(p)) {
                                req_edge = 1;
                                failed = 1;
                            }
                        }
                    }
                    edge_is_requested[idx] = req_edge;
                }
            }
        }
    }

    if (failed)
    {
        fprintf(stderr, "internal error: insufficient precision\n");
        result = -1;
        goto finish;
    }

    /* build the json output */
    {
        int j;
        double d;
        json_t *j_data, *x;
        j_data = json_array();
        for (i = 0; i < r_site->selection_len; i++)
        {
            site = r_site->selection[i];
            for (j = 0; j < r_edge->selection_len; j++)
            {
                edge = r_edge->selection[j];
                idx = m->edge_map->order[edge];
                p = arb_mat_entry(final, site, idx);
                d = arf_get_d(arb_midref(p), ARF_RND_NEAR);
                x = json_pack("[i, i, f]", site, edge, d);
                json_array_append_new(j_data, x);
            }
        }
        j_out = json_pack("{s:[s, s, s], s:o}",
                "columns", "site", "edge", "value",
                "data", j_data);
    }

finish:

    free(site_is_requested);
    likelihood_ws_clear(w, m);

    arb_clear(site_likelihood);
    _arb_vec_clear(derivatives, edge_count);
    arb_mat_clear(final);

    *result_out = result;
    return j_out;
}


json_t *old_arbplf_deriv_run(void *userdata, json_t *root, int *retcode)
{
    json_t *j_out = NULL;
    json_t *model_and_data = NULL;
    json_t *site_reduction = NULL;
    json_t *edge_reduction = NULL;
    int node_count = 0;
    int edge_count = 0;
    int site_count = 0;
    int result = 0;
    model_and_data_t m;
    csr_graph_struct * g = NULL;
    column_reduction_t r_site;
    column_reduction_t r_edge;
    int *idx_to_a = NULL;
    int *b_to_idx = NULL;
    int i, idx;
    int site, edge;
    int *edge_selection_count = NULL;
    int *site_selection_count = NULL;

    model_and_data_init(m);

    column_reduction_init(r_site);
    column_reduction_init(r_edge);

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
                "{s:o, s?o, s?o}",
                "model_and_data", &model_and_data,
                "site_reduction", &site_reduction,
                "edge_reduction", &edge_reduction
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

    /* unpack the tree graph and some counts from the model and data */
    g = m->g;
    site_count = pmat_nsites(m->p);
    edge_count = g->nnz;
    node_count = g->n;

    /* validate the (optional) site reduction section of the json input */
    result = validate_column_reduction(
            r_site, site_count, "site", site_reduction);
    if (result) goto finish;

    /* validate the (optional) edge reduction section of the json input */
    result = validate_column_reduction(
            r_edge, edge_count, "edge", edge_reduction);
    if (result) goto finish;

    /* Indicate which site indices are included in the selection. */
    site_selection_count = calloc(site_count, sizeof(int));
    for (i = 0; i < r_site->selection_len; i++)
    {
        site = r_site->selection[i];
        site_selection_count[site]++;
    }

    /*
     * Indicate which edge indices are included in the selection.
     * Note that the edge_selection_count array uses csr edge indices
     * rather than the user-visible edge ordering.
     */
    edge_selection_count = calloc(edge_count, sizeof(int));
    for (i = 0; i < r_edge->selection_len; i++)
    {
        edge = r_edge->selection[i];
        idx = m->edge_map->order[edge];
        edge_selection_count[idx]++;
    }

    /* Get maps for navigating towards the root of the tree. */
    idx_to_a = malloc(edge_count * sizeof(int));
    b_to_idx = malloc(node_count * sizeof(int));
    _csr_graph_get_backward_maps(idx_to_a, b_to_idx, m->g);

    /*
     * Dispatch to a helper function according to whether or not
     * the user has specified an aggregation of sites or of edges.
     * This is a bit of a hack.
     */
    int agg_site = r_site->agg_mode != AGG_NONE;
    int agg_edge = r_edge->agg_mode != AGG_NONE;
    if (agg_site && agg_edge)
    {
        j_out = _agg_site_edge_yy(
                m, idx_to_a, b_to_idx, r_site, r_edge,
                site_selection_count, edge_selection_count,
                &result);
        if (result) goto finish;
    }
    else if (agg_site && !agg_edge)
    {
        j_out = _agg_site_edge_yn(
                m, idx_to_a, b_to_idx, r_site, r_edge,
                site_selection_count, edge_selection_count,
                &result);
        if (result) goto finish;
    }
    else if (!agg_site && agg_edge)
    {
        j_out = _agg_site_edge_ny(
                m, idx_to_a, b_to_idx, r_site, r_edge,
                site_selection_count, edge_selection_count,
                &result);
        if (result) goto finish;
    }
    else if (!agg_site && !agg_edge)
    {
        j_out = _agg_site_edge_nn(
                m, idx_to_a, b_to_idx, r_site, r_edge,
                site_selection_count, edge_selection_count,
                &result);
        if (result) goto finish;
    }
    else
    {
        fprintf(stderr, "internal error: oops missed a case\n");
        abort();
    }

finish:

    *retcode = result;

    free(site_selection_count);
    free(edge_selection_count);
    column_reduction_clear(r_site);
    column_reduction_clear(r_edge);
    model_and_data_clear(m);

    free(idx_to_a);
    free(b_to_idx);

    flint_cleanup();
    return j_out;
}


static void
_nd_accum_update(nd_accum_t arr,
        likelihood_ws_t w, cross_site_ws_t csw, model_and_data_t m, slong prec)
{
    int site, i, j;
    arb_t cat_lhood, prior_prob, post_lhood, post_lhood_sum;
    nd_axis_struct *site_axis, *edge_axis;
    int *coords;
    slong cat;

    slong ncats = model_and_data_rate_category_count(m);
    slong site_count = model_and_data_site_count(m);

    arb_init(cat_lhood);
    arb_init(prior_prob);
    arb_init(post_lhood);
    arb_init(post_lhood_sum);

    coords = malloc(arr->ndim * sizeof(int));

    site_axis = arr->axes + 0;
    edge_axis = arr->axes + 1;

    /* zero all requested cells of the array */
    nd_accum_zero_requested_cells(arr);

    /*
     * Update the output array at the given precision.
     * Axes have already been updated for this precision.
     * The nd array links to axis selection and aggregation information,
     * and the model and data are provided separately.
     */
    for (site = 0; site < site_count; site++)
    {
        /* skip sites that are not requested */
        if (!site_axis->request_update[site]) continue;
        coords[0] = site;

        /* update base node vectors */
        pmat_update_base_node_vectors(
                w->base_node_vectors, m->p, site,
                m->root_prior, csw->equilibrium,
                m->preorder[0], prec);

        /*
         * Set all cross-category marginal vectors to zero,
         * preparing to accumulate across categories.
         */
        for (i = 0; i < csw->node_count; i++)
        {
            arb_mat_zero(w->cc_marginal_node_vectors + i);
        }

        /*
         * For each category, compute a likelihood for the current site,
         * and compute marginal distributions at all requested nodes.
         */
        arb_zero(post_lhood_sum);
        for (cat = 0; cat < ncats; cat++)
        {
            const arb_mat_struct * tmat_base;
            tmat_base = tmat_collection_entry(csw->transition_matrices, cat, 0);

            pmat_update_base_node_vectors(
                    w->base_node_vectors, m->p, site,
                    m->root_prior, csw->equilibrium,
                    m->preorder[0], prec);

            evaluate_site_lhood(lhood,
                    w->lhood_node_vectors,
                    w->lhood_edge_vectors,
                    w->base_node_vectors,
                    tmat_base,
                    m->g, m->preorder, csw->node_count, prec);

            evaluate_site_derivatives(
                    derivatives,
                    edge_is_requested,
                    m, csw, w,
                    idx_to_a, b_to_idx, site, prec);
            for (idx = 0; idx < edge_count; idx++)
            {
                if (edge_is_requested[idx])
                {
                    arb_div(arb_mat_entry(final, site, idx),
                            derivatives + idx, site_likelihood, prec);
                }
            }

            /*
             * Update per-node and per-edge likelihood vectors.
             * This is a backward pass from the leaves to the root.
             */
            evaluate_site_lhood(cat_lhood,
                    w->lhood_node_vectors,
                    w->lhood_edge_vectors,
                    w->base_node_vectors,
                    tmat_base,
                    m->g, m->preorder, csw->node_count, prec);

            /* Compute the likelihood for the site and category. */
            rate_mixture_get_prob(prior_prob, m->rate_mixture, cat, prec);
            arb_mul(post_lhood, prior_prob, cat_lhood, prec);

            /*
             * If the posterior probability is zero,
             * then ignore marginal probabilities for this category.
             */
            if (arb_is_zero(post_lhood))
                continue;

            arb_add(post_lhood_sum, post_lhood_sum, post_lhood, prec);

            /*
             * Update marginal distribution vectors at nodes.
             * This is a forward pass from the root to the leaves.
             */
            evaluate_site_marginal(
                    w->marginal_node_vectors,
                    w->lhood_node_vectors,
                    w->lhood_edge_vectors,
                    tmat_base,
                    m->g, m->preorder, csw->node_count, csw->state_count, prec);

            /*
             * Accumulate the marginal probabilities for this category.
             */
            for (i = 0; i < csw->node_count; i++)
            {
                arb_mat_struct *a = w->marginal_node_vectors + i;
                arb_mat_struct *b = w->cc_marginal_node_vectors + i;
                for (j = 0; j < csw->state_count; j++)
                {
                    arb_addmul(arb_mat_entry(b, j, 0),
                               arb_mat_entry(a, j, 0), post_lhood, prec);
                }
            }
        }

        /* Divide by the likelihood. */
        for (i = 0; i < csw->node_count; i++)
        {
            arb_mat_scalar_div_arb(
                    w->cc_marginal_node_vectors + i,
                    w->cc_marginal_node_vectors + i,
                    post_lhood_sum, prec);
        }

        /* Update the nd accumulator. */
        for (i = 0; i < csw->node_count; i++)
        {
            arb_mat_struct *nvec;

            /* skip nodes that are not requested */
            if (!node_axis->request_update[i]) continue;
            coords[1] = i;

            nvec = w->cc_marginal_node_vectors + i;
            for (j = 0; j < csw->state_count; j++)
            {
                /* skip states that are not requested */
                if (!state_axis->request_update[j]) continue;
                coords[2] = j;

                /* accumulate */
                nd_accum_accumulate(arr,
                        coords, arb_mat_entry(nvec, j, 0), prec);
            }
        }
    }

    arb_clear(cat_lhood);
    arb_clear(prior_prob);
    arb_clear(post_lhood);
    arb_clear(post_lhood_sum);

    free(coords);
}


static json_t *
_query(model_and_data_t m,
        column_reduction_t r_site,
        column_reduction_t r_edge, int *result_out)
{
    json_t * j_out = NULL;
    slong prec;
    int axis_idx;
    nd_axis_struct axes[3];
    nd_accum_t arr;
    cross_site_ws_t csw;
    likelihood_ws_t w;
    int ndim = 2;
    int result = 0;

    slong site_count = model_and_data_site_count(m);
    slong edge_count = model_and_data_edge_count(m);

    /* initialize likelihood workspace */
    cross_site_ws_pre_init(csw);
    likelihood_ws_init(w, m);

    /* initialize axes at zero precision */
    nd_axis_init(axes+0, "site", site_count, r_site, 0, NULL, 0);
    nd_axis_init(axes+1, "edge", edge_count, r_edge, 0, NULL, 0);

    /* initialize nd accumulation array */
    nd_accum_pre_init(arr);
    nd_accum_init(arr, axes, ndim);

    /* repeat with increasing precision until there is no precision failure */
    int success = 0;
    for (prec=4; !success; prec <<= 1)
    {
        cross_site_ws_reinit(csw, m, prec);

        /* recompute axis reduction weights with increased precision */
        nd_axis_update_precision(axes+0, r_site, prec);
        nd_axis_update_precision(axes+1, r_edge, prec);

        /* recompute the output array with increased working precision */
        _nd_accum_update(arr, w, csw, m, prec);

        /* check whether entries are accurate to full relative precision  */
        success = nd_accum_can_round(arr);
    }

    /* build the json output using the nd array */
    j_out = nd_accum_get_json(arr, &result);
    if (result) goto finish;

finish:

    /* clear likelihood workspace */
    cross_site_ws_clear(csw);
    likelihood_ws_clear(w, m);

    /* clear axes */
    for (axis_idx = 0; axis_idx < ndim; axis_idx++)
    {
        nd_axis_clear(axes + axis_idx);
    }

    /* clear nd accumulation array */
    nd_accum_clear(arr);

    *result_out = result;
    return j_out;
}


static int
_parse(model_and_data_t m,
        column_reduction_t r_site,
        column_reduction_t r_edge, json_t *root)
{
    json_t *model_and_data = NULL;
    json_t *site_reduction = NULL;
    json_t *edge_reduction = NULL;
    slong site_count, edge_count;
    int result = 0;

    /* unpack the top level of json input */
    {
        size_t flags;
        json_error_t err;
        flags = JSON_STRICT;
        result = json_unpack_ex(root, &err, flags,
                "{s:o, s?o, s?o}",
                "model_and_data", &model_and_data,
                "site_reduction", &site_reduction,
                "edge_reduction", &edge_reduction
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
    edge_count = model_and_data_edge_count(m);

    /* validate the site reduction section of the json input */
    result = validate_column_reduction(
            r_site, site_count, "site", site_reduction);
    if (result) return result;

    /* validate the edge reduction section of the json input */
    result = validate_column_reduction(
            r_edge, edge_count, "edge", edge_reduction);
    if (result) return result;

    return result;
}


json_t *arbplf_deriv_run(void *userdata, json_t *root, int *retcode)
{
    json_t *j_out = NULL;
    model_and_data_t m;
    column_reduction_t r_site;
    column_reduction_t r_edge;
    int result = 0;

    model_and_data_init(m);
    column_reduction_init(r_site);
    column_reduction_init(r_edge);

    if (userdata)
    {
        fprintf(stderr, "internal error: unexpected userdata\n");
        result = -1;
        goto finish;
    }

    result = _parse(m, r_site, r_edge, root);
    if (result) goto finish;

    j_out = _query(m, r_site, r_edge, &result);
    if (result) goto finish;

finish:

    *retcode = result;

    model_and_data_clear(m);
    column_reduction_clear(r_site);
    column_reduction_clear(r_edge);

    flint_cleanup();
    return j_out;
}
