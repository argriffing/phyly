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

#include "parsemodel.h"
#include "parsereduction.h"
#include "runjson.h"
#include "arbplfderiv.h"



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
    arb_mat_struct *lhood_edge_column_vectors;
    arb_mat_struct *deriv_node_column_vectors;
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
        w->lhood_edge_column_vectors = NULL;
        w->deriv_node_column_vectors = NULL;
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
    w->lhood_edge_column_vectors = flint_malloc(
            w->edge_count * sizeof(arb_mat_struct));
    w->lhood_node_column_vectors = flint_malloc(
            w->node_count * sizeof(arb_mat_struct));
    w->deriv_node_column_vectors = flint_malloc(
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
    _arb_mat_scalar_div_d(w->rate_matrix, m->rate_divisor, w->prec);

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
     * Workspace for state distributions at nodes.
     * The contents of these vectors can depend on the prior distribution
     * or on data, for example.
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

    /*
     * Allocate an arbitrary precision column vector for each node.
     * This will be used to accumulate derivatives of likelihoods.
     */
    for (i = 0; i < w->node_count; i++)
    {
        nmat = w->deriv_node_column_vectors + i;
        arb_mat_init(nmat, w->state_count, 1);
    }

    /*
     * Allocate an arbitrary precision column vector for each edge.
     * This will be used to cache intermediate values created during
     * the likelihood dynamic programming and which can be reused
     * for derivative calculations.
     * For a given edge of interest, the computation of the derivative
     * with respect to the rate coefficient of that edge can use precomputed
     * per-edge vectors for all edges except the ones on the path from that
     * edge to the root of the tree.
     */
    for (idx = 0; idx < w->edge_count; idx++)
    {
        nmat = w->lhood_edge_column_vectors + idx;
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
        arb_mat_clear(w->lhood_edge_column_vectors + idx);
    }
    flint_free(w->transition_matrices);
    flint_free(w->lhood_edge_column_vectors);
    
    for (i = 0; i < w->node_count; i++)
    {
        arb_mat_clear(w->lhood_node_column_vectors + i);
        arb_mat_clear(w->deriv_node_column_vectors + i);
    }
    flint_free(w->lhood_node_column_vectors);
    flint_free(w->deriv_node_column_vectors);
}


/* helper function to update base node probabilities at a site */
static void
_update_base_node_vectors(
        arb_mat_struct *base_node_vectors,
        pmat_t p, slong site)
{
    slong i, j;
    slong node_count, state_count;
    arb_mat_struct *bvec;
    node_count = pmat_nrows(p);
    state_count = pmat_ncols(p);
    for (i = 0; i < node_count; i++)
    {
        bvec = base_node_vectors + i;
        for (j = 0; j < state_count; j++)
        {
            arb_set_d(arb_mat_entry(bvec, j, 0), *pmat_entry(p, site, i, j));
        }
    }
}


/* Helper function to update likelihood-related vectors at a site. */
static int
_update_lhood_vectors(arb_t lhood,
        model_and_data_t m, likelihood_ws_t w, int site)
{
    _update_base_node_vectors(w->base_node_column_vectors, m->p, site);

    evaluate_site_lhood(lhood,
            w->lhood_node_column_vectors,
            w->lhood_edge_column_vectors,
            w->base_node_column_vectors,
            w->transition_matrices,
            m->g, m->preorder, w->node_count, w->prec);

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
evaluate_site_derivatives(arb_struct *derivatives,
        int *edge_is_requested, model_and_data_t m, likelihood_ws_t w,
        int *idx_to_a, int *b_to_idx, int site)
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

    g = m->g;
    _arb_vec_zero(derivatives, w->edge_count);

    /*
     * For each requested edge at this site,
     * compute the state vectors associated with the derivative.
     * This involves tracing back from the edge to the root.
     */
    int deriv_idx;
    int curr_idx;
    for (deriv_idx = 0; deriv_idx < w->edge_count; deriv_idx++)
    {
        if (!edge_is_requested[deriv_idx])
            continue;

        curr_idx = deriv_idx;
        while (curr_idx != -1)
        {
            a = idx_to_a[curr_idx];

            /* initialize the state vector for node a */
            nmat = w->deriv_node_column_vectors + a;
            for (state = 0; state < w->state_count; state++)
            {
                tmpd = *pmat_entry(m->p, site, a, state);
                arb_set_d(arb_mat_entry(nmat, state, 0), tmpd);
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
                    rmat = w->rate_matrix;
                    emat = w->lhood_edge_column_vectors + idx;
                    _prune_update(nmat, nmat, rmat, emat, w->prec);
                }
                else if (idx == curr_idx)
                {
                    b = g->indices[idx];
                    tmat = w->transition_matrices + idx;
                    nmatb = w->deriv_node_column_vectors + b;
                    _prune_update(nmat, nmat, tmat, nmatb, w->prec);
                }
                else
                {
                    emat = w->lhood_edge_column_vectors + idx;
                    _arb_mat_mul_entrywise(nmat, nmat, emat, w->prec);
                }
            }

            /* move back towards the root of the tree */
            curr_idx = b_to_idx[a];
        }

        /* Report the sum of state entries associated with the root. */
        nmat = w->deriv_node_column_vectors + m->root_node_index;
        arb_struct * deriv = derivatives + deriv_idx;
        arb_zero(deriv);
        for (state = 0; state < w->state_count; state++)
        {
            arb_add(deriv, deriv, arb_mat_entry(nmat, state, 0), w->prec);
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

    likelihood_ws_init(w, NULL, 0);

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

        likelihood_ws_clear(w);
        likelihood_ws_init(w, m, prec);

        /* clear the accumulation across sites */
        arb_zero(deriv_ll_site_accum);

        for (site = 0; site < site_count; site++)
        {
            if (!site_selection_count[site])
                continue;

            result = _update_lhood_vectors(site_likelihood, m, w, site);
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
                    m, w,
                    idx_to_a, b_to_idx, site);

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

    likelihood_ws_clear(w);
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

    likelihood_ws_init(w, NULL, 0);

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

        likelihood_ws_clear(w);
        likelihood_ws_init(w, m, prec);

        /* clear the accumulation across sites */
        _arb_vec_zero(deriv_ll_accum, edge_count);

        for (site = 0; site < site_count; site++)
        {
            if (!site_selection_count[site])
                continue;

            result = _update_lhood_vectors(site_likelihood, m, w, site);
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
                    m, w,
                    idx_to_a, b_to_idx, site);

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
    likelihood_ws_clear(w);

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

    likelihood_ws_init(w, NULL, 0);

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

        likelihood_ws_clear(w);
        likelihood_ws_init(w, m, prec);

        /* update edge derivative accumulations at requested sites */
        for (site = 0; site < site_count; site++)
        {
            if (!site_is_requested[site])
                continue;

            result = _update_lhood_vectors(site_likelihood, m, w, site);
            if (result) goto finish;

            evaluate_site_derivatives(
                    derivatives,
                    edge_selection_count,
                    m, w,
                    idx_to_a, b_to_idx, site);
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
    likelihood_ws_clear(w);

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

    likelihood_ws_init(w, NULL, 0);

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
        likelihood_ws_clear(w);
        likelihood_ws_init(w, m, prec);

        /* update requested sites */
        for (site = 0; site < site_count; site++)
        {
            if (!site_is_requested[site])
                continue;

            result = _update_lhood_vectors(site_likelihood, m, w, site);
            if (result) goto finish;

            evaluate_site_derivatives(
                    derivatives,
                    edge_is_requested,
                    m, w,
                    idx_to_a, b_to_idx, site);
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
    likelihood_ws_clear(w);

    arb_clear(site_likelihood);
    _arb_vec_clear(derivatives, edge_count);
    arb_mat_clear(final);

    *result_out = result;
    return j_out;
}


json_t *arbplf_deriv_run(void *userdata, json_t *root, int *retcode)
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
