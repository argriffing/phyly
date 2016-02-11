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

#include "parsemodel.h"
#include "parsereduction.h"
#include "runjson.h"
#include "arbplfderiv.h"

static int _can_round(arb_t x)
{
    return arb_can_round_arf(x, 53, ARF_RND_NEAR);
}

/*
 * Look at edges of the csr graph of the phylogenetic tree,
 * tracking edge_index->initial_node_index and final_node_index->edge_index.
 * Note that the edges do not need to be traversed in any particular order.
 */
static void
_csr_graph_get_backward_maps(int *idx_to_a, int *b_to_idx, csr_graph_t g)
{
    int idx;
    int a, b;
    int node_count;
    node_count = g->n;
    for (b = 0; b < node_count; b++)
    {
        b_to_idx[b] = -1;
    }
    for (a = 0; a < node_count; a++)
    {
        for (idx = g->indptr[a]; idx < g->indptr[a+1]; idx++)
        {
            b = g->indices[idx];
            idx_to_a[idx] = a;
            b_to_idx[b] = idx;
        }
    }
}


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
    for (j = 0; j < w->state_count; j++)
    {
        for (k = 0; k < w->state_count; k++)
        {
            double r;
            r = *dmat_entry(m->mat, j, k);
            arb_set_d(arb_mat_entry(w->rate_matrix, j, k), r);
        }
    }

    /*
     * Modify the diagonals of the unscaled rate matrix
     * so that the sum of each row is zero.
     */
    arb_ptr p;
    for (j = 0; j < w->state_count; j++)
    {
        p = arb_mat_entry(w->rate_matrix, j, j);
        arb_zero(p);
        for (k = 0; k < w->state_count; k++)
        {
            if (j != k)
            {
                arb_sub(p, p, arb_mat_entry(w->rate_matrix, j, k), w->prec);
            }
        }
    }

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


static void
_arb_mat_mul_entrywise(arb_mat_t c, arb_mat_t a, arb_mat_t b, slong prec)
{
    slong i, j, nr, nc;

    nr = arb_mat_nrows(a);
    nc = arb_mat_ncols(a);

    for (i = 0; i < nr; i++)
    {
        for (j = 0; j < nc; j++)
        {
            arb_mul(arb_mat_entry(c, i, j),
                    arb_mat_entry(a, i, j),
                    arb_mat_entry(b, i, j), prec);
        }
    }
}


static void
_prune_update(arb_mat_t d, arb_mat_t c, arb_mat_t a, arb_mat_t b, slong prec)
{
    /*
     * d = c o a*b
     * Analogous to _arb_mat_addmul
     * except with entrywise product instead of entrywise addition.
     */
    slong m, n;

    m = a->r;
    n = b->c;

    arb_mat_t tmp;
    arb_mat_init(tmp, m, n);

    arb_mat_mul(tmp, a, b, prec);
    _arb_mat_mul_entrywise(d, c, tmp, prec);
    arb_mat_clear(tmp);
}


/* Calculate the likelihood, storing many intermediate calculations. */
static void
evaluate_site_likelihood(arb_t lhood,
        model_and_data_t m, likelihood_ws_t w, int site)
{
    int u, a, b, idx;
    int start, stop;
    int state;
    double tmpd;
    arb_mat_struct * nmat;
    arb_mat_struct * nmatb;
    arb_mat_struct * tmat;
    arb_mat_struct * emat;
    csr_graph_struct *g;

    g = m->g;

    /*
     * Fill all of the per-node and per-edge likelihood-related vectors.
     * Note that because edge derivatives are requested,
     * the vectors on edges are stored explicitly.
     * In the likelihood-only variant of this function, these per-edge
     * vectors are temporary variables whose lifespan is only long enough
     * to update the vector associated with the parent node of the edge.
     */
    for (u = 0; u < w->node_count; u++)
    {
        a = m->preorder[w->node_count - 1 - u];
        start = g->indptr[a];
        stop = g->indptr[a+1];

        /* create all of the state vectors on edges outgoing from this node */
        for (idx = start; idx < stop; idx++)
        {
            b = g->indices[idx];
            /*
             * At this point (a, b) is an edge from node a to node b
             * in a post-order traversal of edges of the tree.
             */
            tmat = w->transition_matrices + idx;
            nmatb = w->lhood_node_column_vectors + b;
            emat = w->lhood_edge_column_vectors + idx;
            arb_mat_mul(emat, tmat, nmatb, w->prec);
        }

        /* initialize the state vector for node a */
        nmat = w->lhood_node_column_vectors + a;
        for (state = 0; state < w->state_count; state++)
        {
            tmpd = *pmat_entry(m->p, site, a, state);
            arb_set_d(arb_mat_entry(nmat, state, 0), tmpd);
        }

        /* multiplicatively accumulate state vectors at this node */
        for (idx = start; idx < stop; idx++)
        {
            emat = w->lhood_edge_column_vectors + idx;
            _arb_mat_mul_entrywise(nmat, nmat, emat, w->prec);
        }
    }

    /* Report the sum of state entries associated with the root. */
    int root_node_index = m->preorder[0];
    nmat = w->lhood_node_column_vectors + root_node_index;
    arb_zero(lhood);
    for (state = 0; state < w->state_count; state++)
    {
        arb_add(lhood, lhood, arb_mat_entry(nmat, state, 0), w->prec);
    }
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
        int *edge_deriv_is_requested, model_and_data_t m, likelihood_ws_t w,
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
        if (!edge_deriv_is_requested[deriv_idx])
        {
            continue;
        }
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


static int
_get_site_agg_weights(
        arb_t weight_divisor, arb_struct * weights,
        model_and_data_t m, column_reduction_t r, slong prec)
{
    int i, site;
    int * site_selection_count = NULL;
    int result = 0;
    int site_count = pmat_nsites(m->p);

    _arb_vec_zero(weights, site_count);

    /* define site selection count if necessary */
    if (r->agg_mode == AGG_SUM || r->agg_mode == AGG_AVG)
    {
        site_selection_count = calloc(site_count, sizeof(int));
        for (i = 0; i < r->selection_len; i++)
        {
            site = r->selection[i];
            site_selection_count[site]++;
        }
    }

    if (r->agg_mode == AGG_WEIGHTED_SUM)
    {
        arb_t weight;
        arb_init(weight);
        for (i = 0; i < r->selection_len; i++)
        {
            site = r->selection[i];
            arb_set_d(weight, r->weights[i]);
            arb_add(weights+site, weights+site, weight, prec);
        }
        arb_clear(weight);
        arb_one(weight_divisor);
    }
    else if (r->agg_mode == AGG_SUM)
    {
        for (site = 0; site < site_count; site++)
        {
            arb_set_si(weights+site, site_selection_count[site]);
        }
        arb_one(weight_divisor);
    }
    else if (r->agg_mode == AGG_AVG)
    {
        for (site = 0; site < site_count; site++)
        {
            arb_set_si(weights+site, site_selection_count[site]);
        }
        arb_set_si(weight_divisor, r->selection_len);
    }
    else
    {
        fprintf(stderr, "internal error: unexpected aggregation mode\n");
        result = -1;
        goto finish;
    }

finish:

    free(site_selection_count);
    return result;
}


/*
 * Given the user-provided edge reduction,
 * define an edge aggregation weight vector
 * whose indices are with respect to csr tree edges.
 * Also define a weight divisor that applies to all weights.
 */
static int
_get_edge_agg_weights(
        arb_t weight_divisor, arb_struct * weights,
        model_and_data_t m, column_reduction_t r, slong prec)
{
    int i, edge, idx;
    int edge_count = 0;
    int * edge_selection_count = NULL;
    int result = 0;

    edge_count = m->g->nnz;

    _arb_vec_zero(weights, edge_count);

    /* define edge selection count if necessary */
    if (r->agg_mode == AGG_SUM || r->agg_mode == AGG_AVG)
    {
        edge_selection_count = calloc(edge_count, sizeof(int));
        for (i = 0; i < r->selection_len; i++)
        {
            edge = r->selection[i];
            idx = m->edge_map->order[edge];
            edge_selection_count[idx]++;
        }
    }

    if (r->agg_mode == AGG_WEIGHTED_SUM)
    {
        arb_t weight;
        arb_init(weight);
        for (i = 0; i < r->selection_len; i++)
        {
            edge = r->selection[i];
            arb_set_d(weight, r->weights[i]);
            idx = m->edge_map->order[edge];
            arb_add(weights+idx, weights+idx, weight, prec);
        }
        arb_clear(weight);
        arb_one(weight_divisor);
    }
    else if (r->agg_mode == AGG_SUM)
    {
        for (idx = 0; idx < edge_count; idx++)
        {
            arb_set_si(weights+idx, edge_selection_count[idx]);
        }
        arb_one(weight_divisor);
    }
    else if (r->agg_mode == AGG_AVG)
    {
        for (idx = 0; idx < edge_count; idx++)
        {
            arb_set_si(weights+idx, edge_selection_count[idx]);
        }
        arb_set_si(weight_divisor, r->selection_len);
    }
    else
    {
        fprintf(stderr, "internal error: unexpected aggregation mode\n");
        result = -1;
        goto finish;
    }

finish:

    free(edge_selection_count);
    return result;
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
        result = _get_edge_agg_weights(
                edge_weight_divisor, edge_weights, m, r_edge, prec);
        if (result) goto finish;

        /* define site aggregation weights */
        result = _get_site_agg_weights(
                site_weight_divisor, site_weights, m, r_site, prec);
        if (result) goto finish;

        likelihood_ws_clear(w);
        likelihood_ws_init(w, m, prec);

        /* clear the accumulation across sites */
        arb_zero(deriv_ll_site_accum);

        for (site = 0; site < site_count; site++)
        {
            /* skip sites that are not selected */
            if (!site_selection_count[site])
            {
                continue;
            }

            /*
             * Evaluate the site likelihood and compute the per-node and
             * per-edge vectors that will be reused for computing
             * the derivatives.
             */
            evaluate_site_likelihood(site_likelihood, m, w, site);

            /*
             * If any site likelihood is exactly zero
             * then we do not need to continue.
             * todo: if the site likelihood interval includes zero
             *       then this could be handled at this point,
             *       for example by immediately requesting higher precision
             */
            if (arb_is_zero(site_likelihood))
            {
                fprintf(stderr, "error: infeasible\n");
                result = -1;
                goto finish;
            }

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

    int *edge_deriv_is_requested = NULL;

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
    edge_deriv_is_requested = malloc(edge_count * sizeof(int));
    for (idx = 0; idx < edge_count; idx++)
    {
        edge_deriv_is_requested[idx] = edge_selection_count[idx] != 0;
    }

    /* repeat with increasing precision until there is no precision failure */
    for (failed=1, prec=4; failed; prec<<=1)
    {
        /* define site aggregation weights */
        result = _get_site_agg_weights(
                site_weight_divisor, site_weights, m, r_site, prec);
        if (result) goto finish;

        likelihood_ws_clear(w);
        likelihood_ws_init(w, m, prec);

        /* clear the accumulation across sites */
        _arb_vec_zero(deriv_ll_accum, edge_count);

        for (site = 0; site < site_count; site++)
        {
            /* skip sites that are not selected */
            if (!site_selection_count[site])
            {
                continue;
            }

            /*
             * Evaluate the site likelihood and compute the per-node and
             * per-edge vectors that will be reused for computing
             * the derivatives.
             */
            evaluate_site_likelihood(site_likelihood, m, w, site);

            /*
             * If any site likelihood is exactly zero
             * then we do not need to continue.
             * todo: if the site likelihood interval includes zero
             *       then this could be handled at this point,
             *       for example by immediately requesting higher precision
             */
            if (arb_is_zero(site_likelihood))
            {
                fprintf(stderr, "error: infeasible\n");
                result = -1;
                goto finish;
            }

            /*
             * Evaluate derivatives of site likelihood
             * with respect to each selected edge.
             * Note that these derivatives are with respect to the likelihood
             * rather than with respect to the log likelihood.
             */
            evaluate_site_derivatives(
                    derivatives,
                    edge_deriv_is_requested,
                    m, w,
                    idx_to_a, b_to_idx, site);

            /* define a site-specific coefficient */
            arb_div(x, site_weights+site, site_weight_divisor, prec);
            arb_div(x, x, site_likelihood, prec);

            /* accumulate */
            for (idx = 0; idx < edge_count; idx++)
            {
                if (edge_deriv_is_requested[idx])
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
            if (edge_deriv_is_requested[idx])
            {
                if (_can_round(deriv_ll_accum + idx))
                {
                    /*
                    printf("debug: can round idx=%d prec=%d\n",
                            idx, (int) prec);
                    */
                    edge_deriv_is_requested[idx] = 0;
                    arb_set(final + idx, deriv_ll_accum + idx);
                }
                else
                {
                    /*
                    printf("debug: cannot round idx=%d prec=%d\n",
                            idx, (int) prec);
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

    free(edge_deriv_is_requested);
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
    /*
    else if (!agg_site && agg_edge)
    {
        j_out = _agg_site_edge_ny(&result);
        if (result) goto finish;
    }
    else if (!agg_site && !agg_edge)
    {
        j_out = _agg_site_edge_nn(&result);
        if (result) goto finish;
    }
    else
    {
        fprintf(stderr, "internal error: oops missed a case\n");
        abort();
    }
    */
    else
    {
        fprintf(stderr, "aggregation mode not yet implemented\n");
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
