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
    int edge_count;
    int node_count;
    edge_count = g->nnz;
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
    for (i = 0; i < w->edge_count; i++)
    {
        nmat = w->deriv_edge_column_vectors + i;
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
        arb_mat_clear(w->deriv_edge_column_vectors + i);
    }
    flint_free(w->transition_matrices);
    flint_free(w->deriv_edge_column_vectors);
    
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
_arb_mat_addmul(arb_mat_t d, arb_mat_t c, arb_mat_t a, arb_mat_t b, slong prec)
{
    /*
     * d = c + a*b
     * c and d may be aliased with each other but not with a or b
     * implementation inspired by @aaditya-thakkar's flint2 pull request
     * https://github.com/wbhart/flint2/pull/215
     *
     * todo: it should be possible to do something like this
     *       without making a temporary array
     */

    slong m, n;

    m = a->r;
    n = b->c;

    arb_mat_t tmp;
    arb_mat_init(tmp, m, n);

    arb_mat_mul(tmp, a, b, prec);
    arb_mat_add(d, c, tmp, prec);
    arb_mat_clear(tmp);
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


static void
_arb_vec_printd(arb_struct *v, slong n, slong d)
{
    int i;
    flint_printf("[");
    for (i = 0; i < n; i++)
    {
        if (i) flint_printf(", ");
        arb_printd(v+i, d);
    }
    flint_printf("]");
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
    arb_mat_struct * rmat;
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
        nmat = w->node_column_vectors + a;
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
    nmat = w->node_column_vectors + root_node_index;
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
 * and for indexing the edge_is_selected array.
 *
 * The idx_to_a and b_to_idx arrays define the maps
 * edge->initial_node and final_node->edge respectively.
 */
static void
evaluate_site_derivatives(arb_struct *derivatives,
        int *edge_is_selected, model_and_data_t m, likelihood_ws_t w,
        int *idx_to_a, int *b_to_idx, int site)
{
    int u, a, b, idx;
    int start, stop;
    int state;
    double tmpd;
    arb_mat_struct * nmat;
    arb_mat_struct * nmatb;
    arb_mat_struct * tmat;
    arb_mat_struct * rmat;
    csr_graph_struct *g;

    g = m->g;

    /*
     * For each requested edge at this site,
     * compute the state vectors associated with the derivative.
     * This involves tracing back from the edge to the root.
     */
    int deriv_idx;
    int curr_idx;
    for (deriv_idx = 0; deriv_idx < w->edge_count; deriv_idx++)
    {
        if (!edge_is_selected[deriv_idx])
        {
            continue;
        }
        curr_idx = deriv_idx;
        a = idx_to_a[curr_idx];
        while (a != -1)
        {
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
                    emat = w->lhood_edge_column_vectors + idx;
                    rmat = w->rate_matrix;
                    _prune_update(nmat, nmat, rmat, emat, w->prec);
                }
                else if (idx == curr_idx)
                {
                    b = g->indices[idx];
                    nmatb = w->deriv_node_column_vectors + b;
                    tmat = w->transition_matrices + idx;
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
            a = idx_to_a[curr_idx];
        }

        /* Report the sum of state entries associated with the root. */
        nmat = w->deriv_node_column_vectors + m->root_node_index;
        arb_struct * deriv;
        arb_zero(derivatives + deriv_idx);
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
        int *site_is_selected, int *edge_is_selected, int *result_out)
{
    json_t * out;
    likelihood_ws_t w;
    arb_t accum;
    arb_struct * likelihoods;
    arb_struct * derivatives;
    int site;

    /*
     * The recipe is:
     * 1) repeat with increasing precision until arb gives the green light:
     *    2) for each site
     *       2a) aggregate derivatives of likelihood across edges
     *       2b) divide the aggregate by the likelihood, thereby
     *           converting from derivative of likelihood to derivative
     *           of log likelihood
     *    3) aggregate those aggregates across sites
     *
     * Note that this uses the linearity of the aggregate functions
     * in a nontrivial way in step (2b).
     */

    evaluate_site_derivatives(
            derivatives,
            edge_is_selected,
            m, w,
            idx_to_a, b_to_idx, site);

    *result_out = 0;
    return NULL;
}


json_t *arbplf_deriv_run(void *userdata, json_t *root, int *retcode)
{
    json_t *j_out = NULL;
    json_t *model_and_data = NULL;
    json_t *site_reduction = NULL;
    int edge_count = 0;
    int site_count = 0;
    int node_count = 0;
    int result = 0;
    model_and_data_t m;
    int *site_is_selected = NULL;
    arb_struct * site_likelihoods = NULL;
    arb_t aggregate, tmp, weight;
    likelihood_ws_t w;
    int iter = 0;
    int i;
    int edge, idx, site;
    csr_graph_struct * g = NULL;
    column_reduction_t r_site;
    column_reduction_t r_edge;

    arb_init(tmp);
    arb_init(weight);
    arb_init(aggregate);

    likelihood_ws_init(w, NULL, 0);
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

    /* unpack the tree graph and some counts from the model and data */
    g = m->g;
    site_count = pmat_nsites(m->p);
    edge_count = g->nnz;
    node_count = g->n;

    /* validate the (optional) site reduction section of the json input */
    result = validate_column_reduction(
            r_site, site_count, "site", site_reduction);
    if (result) goto finish;

    /* Indicate which site indices are included in the selection. */
    site_is_selected = calloc(site_count, sizeof(int));
    for (i = 0; i < r_site->selection_len; i++)
    {
        site = r_site->selection[i];
        site_is_selected[site] = 1;
    }

    /*
     * Indicate which edge indices are included in the selection.
     * Note that the edge_is_selected array uses csr edge indices
     * rather than the user-visible edge ordering.
     */
    edge_is_selected = calloc(edge_count, sizeof(int));
    for (i = 0; i < r_edge->selection_len; i++)
    {
        edge = r_edge->selection[i];
        idx = m->edge_mapper->order[edge];
        edge_is_selected[idx] = 1;
    }

    /*
     * Dispatch to a helper function according to whether or not
     * the user has specified an aggregation of sites or of edges.
     * This is a bit of a hack.
     */
    int agg_site = r_site->agg_mode != AGG_NONE;
    int agg_edge = r_edge->agg_mode != AGG_NONE;
    if (agg_site && agg_edge)
    {
        j_out = _agg_site_edge_yy(&result);
        if (result) goto finish;
    }
    else if (agg_site && !agg_edge)
    {
        j_out = _agg_site_edge_yn(&result);
        if (result) goto finish;
    }
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

    site_likelihoods = _arb_vec_init(site_count);

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

            /* if the site is not in the selection then skip it */
            if (!site_is_selected[site])
            {
                continue;
            }

            /* if the log likelihood is fully evaluated then skip it */
            if (iter && r_site->agg_mode == AGG_NONE && _can_round(ll))
            {
                continue;
            }

            evaluate_site_likelihood(lhood, m, w, site);
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
            arb_zero(aggregate);
            arb_one(weight);
            for (i = 0; i < r_site->selection_len; i++)
            {
                site = r_site->selection[i];
                if (r_site->agg_mode == AGG_WEIGHTED_SUM)
                {
                    arb_set_d(weight, r_site->weights[i]);
                }
                arb_addmul(aggregate, ll, weight, prec);
            }
            if (r_site->agg_mode == AGG_AVG)
            {
                arb_div_si(aggregate, aggregate, r_site->selection_len, prec);
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

    if (r_site->agg_mode == AGG_NONE)
    {
        double d;
        json_t *j_data, *x;
        j_data = json_array();
        for (i = 0; i < r_site->selection_len; i++)
        {
            site = r_site->selection[i];
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
    column_reduction_clear(r_site);
    model_and_data_clear(m);
    likelihood_ws_clear(w);
    flint_cleanup();
    return j_out;
}
