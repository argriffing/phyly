/*
 * Use arbitrary precision matrix operations to compute functions related
 * to second order approximations of phylogenetic log likelihood functions.
 *
 * These functions include:
 * 1) computing the hessian matrix
 * 2) computing the inverse of the hessian matrix
 * 3) computing the newton delta
 * 4) computing the newton point
 *
 * The JSON format is used for both input and output.
 * Arbitrary precision is used only internally;
 * double precision floating point without error bounds
 * is used for input and output.
 *
 * See the arbplfll.c or arbplfderiv.c comments (or even the docs if they have
 * been written by now...) for more details.
 *
 * For now, site aggregation is required.
 * For now, edge reduction is forbidden.
 *
 * input format:
 * {
 * "model_and_data" : {
 *  "edges" : [[a, b], [c, d], ...],                 (#edges, 2)
 *  "edge_rate_coefficients" : [a, b, ...],          (#edges, )
 *  "rate_matrix" : [[a, b, ...], [c, d, ...], ...], (#states, #states)
 *  "probability_array" : [...]                      (#sites, #nodes, #states)
 * },
 * "site_reduction" : {
 *  "selection" : [a, b, c, ...], (optional)
 *  "aggregation" : {"sum" | "avg" | [a, b, c, ...]}
 * }}
 *
 * output format (for functions giving edge vectors):
 * {
 *  "columns" : ["edge", "value"],
 *  "data" : [[a, b], [d, e], ..., [x, y]]
 * }
 *
 * output format (for functions giving edge-edge matrices):
 * {
 *  "columns" : ["first_edge", "second_edge", "value"],
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

#include "parsemodel.h"
#include "parsereduction.h"
#include "runjson.h"
#include "arbplfhess.h"


/* this struct does not own its vectors */
typedef struct
{
    arb_mat_struct **node_pvectors;
    arb_mat_struct **edge_pvectors;
} indirect_plane_struct;
typedef indirect_plane_struct indirect_plane_t[1];

static void
indirect_plane_pre_init(indirect_plane_t p)
{
    p->node_pvectors = NULL;
    p->edge_pvectors = NULL;
}

static void
indirect_plane_init(indirect_plane_t p, int node_count, int edge_count)
{
    p->node_pvectors = flint_malloc(node_count * sizeof(arb_mat_struct *));
    p->edge_pvectors = flint_malloc(edge_count * sizeof(arb_mat_struct *));
}

static void
indirect_plane_clear(indirect_plane_t p)
{
    flint_free(p->node_pvectors);
    flint_free(p->edge_pvectors);
}



typedef struct
{
    arb_mat_struct *node_vectors;
    arb_mat_struct *edge_vectors;
    int node_count;
    int edge_count;
} plane_struct;
typedef plane_struct plane_t[1];

static void
plane_pre_init(plane_t p)
{
    p->node_count = 0;
    p->edge_count = 0;
    p->node_vectors = NULL;
    p->edge_vectors = NULL;
}

static void
plane_init(plane_t p, int node_count, int edge_count, int state_count)
{
    int i;
    p->node_count = node_count;
    p->edge_count = edge_count;
    p->node_vectors = flint_malloc(node_count * sizeof(arb_mat_struct));
    p->edge_vectors = flint_malloc(edge_count * sizeof(arb_mat_struct));
    for (i = 0; i < node_count; i++)
    {
        arb_mat_init(p->node_vectors + i, state_count, 1);
    }
    for (i = 0; i < edge_count; i++)
    {
        arb_mat_init(p->edge_vectors + i, state_count, 1);
    }
}

static void
plane_clear(plane_t p)
{
    int i;
    for (i = 0; i < p->node_count; i++)
    {
        arb_mat_clear(p->node_vectors + i);
    }
    for (i = 0; i < p->edge_count; i++)
    {
        arb_mat_clear(p->edge_vectors + i);
    }
    flint_free(p->node_vectors);
    flint_free(p->edge_vectors);
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
    plane_t base_plane;
    plane_t lhood_plane;
    plane_t deriv_plane;
    plane_t hess_plane;
    indirect_plane_t indirect_plane;
} likelihood_ws_struct;
typedef likelihood_ws_struct likelihood_ws_t[1];

static void
likelihood_ws_init(likelihood_ws_t w, model_and_data_t m, slong prec)
{
    csr_graph_struct *g;
    int i, j, k;
    arb_mat_struct * tmat;
    double tmpd;

    if (!m)
    {
        plane_pre_init(w->base_plane);
        plane_pre_init(w->lhood_plane);
        plane_pre_init(w->deriv_plane);
        plane_pre_init(w->hess_plane);
        indirect_plane_pre_init(w->indirect_plane);
        arb_mat_init(w->rate_matrix, 0, 0);
        w->transition_matrices = NULL;
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

    /*
     * Initialize the unscaled arbitrary precision rate matrix.
     * Modify the diagonals so that the sum of each row is zero.
     */
    dmat_get_arb_mat(w->rate_matrix, m->mat);
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

    plane_init(w->base_plane, w->node_count, w->edge_count, w->state_count);
    plane_init(w->lhood_plane, w->node_count, w->edge_count, w->state_count);
    plane_init(w->deriv_plane, w->node_count, w->edge_count, w->state_count);
    plane_init(w->hess_plane, w->node_count, w->edge_count, w->state_count);

    indirect_plane_init(w->indirect_plane, w->node_count, w->edge_count);
}

static void
likelihood_ws_clear(likelihood_ws_t w)
{
    int idx;

    plane_clear(w->base_plane);
    plane_clear(w->lhood_plane);
    plane_clear(w->deriv_plane);
    plane_clear(w->hess_plane);
    indirect_plane_clear(w->indirect_plane);

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
}


/* Calculate the likelihood, storing many intermediate calculations. */
static void
evaluate_site_likelihood(arb_t lhood,
        csr_graph_struct *g, int *preorder,
        likelihood_ws_t w)
{
    int u, a, b, idx;
    int start, stop;
    arb_mat_struct * nmat;
    arb_mat_struct * nmatb;
    arb_mat_struct * tmat;
    arb_mat_struct * emat;
    plane_struct * plane;

    plane = w->lhood_plane;

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
        a = preorder[w->node_count - 1 - u];
        nmat = plane->node_vectors + a;
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
            nmatb = plane->node_vectors + b;
            emat = plane->edge_vectors + idx;
            arb_mat_mul(emat, tmat, nmatb, w->prec);
        }

        /* initialize the state vector for node a */
        arb_mat_set(nmat, w->base_plane->node_vectors + a);

        /* multiplicatively accumulate state vectors at this node */
        for (idx = start; idx < stop; idx++)
        {
            emat = plane->edge_vectors + idx;
            _arb_mat_mul_entrywise(nmat, nmat, emat, w->prec);
        }
    }

    /* Report the sum of state entries associated with the root. */
    int root_node_index = preorder[0];
    nmat = plane->node_vectors + root_node_index;
    _arb_mat_sum(lhood, nmat, w->prec);
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
evaluate_site_derivatives(arb_t derivative,
        csr_graph_struct *g, int *preorder,
        likelihood_ws_t w, int *idx_to_a, int *b_to_idx,
        int deriv_idx, plane_t plane, int update_indirect)
{
    int a, b, idx, curr_idx;
    int start, stop;
    arb_mat_struct * nmat;
    arb_mat_struct * nmatb;
    arb_mat_struct * tmat;
    arb_mat_struct * emat;
    arb_mat_struct * emati;
    arb_mat_struct * rmat;
    indirect_plane_struct * indirect;

    rmat = w->rate_matrix;
    indirect = w->indirect_plane;

    curr_idx = deriv_idx;
    while (curr_idx != -1)
    {
        a = idx_to_a[curr_idx];
        nmat = plane->node_vectors + a;
        start = g->indptr[a];
        stop = g->indptr[a+1];

        /* initialize the state vector for node a */
        arb_mat_set(nmat, w->base_plane->node_vectors + a);

        for (idx = start; idx < stop; idx++)
        {
            if (idx == deriv_idx)
            {
                emati = indirect->edge_pvectors[idx];
                emat = plane->edge_vectors + idx;
                arb_mat_mul(emat, rmat, emati, w->prec);
                _arb_mat_mul_entrywise(nmat, nmat, emat, w->prec);
                if (update_indirect)
                {
                    indirect->edge_pvectors[idx] = emat;
                    indirect->node_pvectors[a] = nmat;
                }
            }
            else if (idx == curr_idx)
            {
                b = g->indices[idx];
                tmat = w->transition_matrices + idx;
                nmatb = plane->node_vectors + b;
                emat = plane->edge_vectors + idx;
                arb_mat_mul(emat, tmat, nmatb, w->prec);
                _arb_mat_mul_entrywise(nmat, nmat, emat, w->prec);
                if (update_indirect)
                {
                    indirect->edge_pvectors[idx] = emat;
                    indirect->node_pvectors[a] = nmat;
                }
            }
            else
            {
                emati = indirect->edge_pvectors[idx];
                _arb_mat_mul_entrywise(nmat, nmat, emati, w->prec);
                if (update_indirect)
                {
                    indirect->node_pvectors[a] = nmat;
                }
            }
        }

        /* move back towards the root of the tree */
        curr_idx = b_to_idx[a];
    }

    /* Report the sum of state entries associated with the root. */
    _arb_mat_sum(derivative, plane->node_vectors + preorder[0], w->prec);
}


/*
 * Second order log likelihood computations aggregated over sites.
 * The calculations use a specified working precision.
 */
static int
_compute_second_order(
        second_order_t so,
        model_and_data_t m, column_reduction_t r_site,
        int *site_selection_count, slong prec)
{
    likelihood_ws_t w;
    int result = 0;
    int i, j, idx, site, state, a;

    int site_count = pmat_nsites(m->p);
    int edge_count = m->g->nnz;
    int node_count = m->g->n;
    int state_count = arb_mat_nrows(m->mat);

    int *idx_to_a = NULL;
    int *b_to_idx = NULL;

    arb_t x;
    arb_t lhood;
    arb_struct * lhood_gradient;
    arb_mat_t lhood_hessian;
    arb_mat_t ll_hessian;

    arb_init(x);
    arb_init(lhood);
    lhood_gradient = _arb_vec_init(edge_count);
    arb_mat_init(lhood_hessian, edge_count, edge_count);
    arb_mat_init(ll_hessian, edge_count, edge_count);

    arb_struct * site_weights;
    arb_t site_weight_divisor;

    arb_init(site_weight_divisor);
    site_weights = _arb_vec_init(site_count);

    /* Get maps for navigating towards the root of the tree. */
    idx_to_a = malloc(edge_count * sizeof(int));
    b_to_idx = malloc(node_count * sizeof(int));
    _csr_graph_get_backward_maps(idx_to_a, b_to_idx, m->g);

    /* define site aggregation weights */
    result = get_site_agg_weights(
            site_weight_divisor, site_weights, site_count, r_site, prec);
    if (result) goto finish;

    likelihood_ws_init(w, m, prec);

    /*
     * Initialize the outputs to zero.
     * These will accumulate across sites.
     */
    second_order_zero(so);

    for (site = 0; site < site_count; site++)
    {
        /* skip sites that are not selected */
        if (!site_selection_count[site])
        {
            continue;
        }

        /*
         * Initialize base plane nodes for this site
         * according to prior state distributions and data.
         * Edges remain unused.
         */
        for (a = 0; a < w->node_count; a++)
        {
            arb_mat_struct * mat;
            mat = w->base_plane->node_vectors + a;
            for (state = 0; state < w->state_count; state++)
            {
                arb_set_d(
                        arb_mat_entry(mat, state, 0),
                        *pmat_entry(m->p, site, a, state));
            }
        }

        /* Reset pointers in the virtual plane. */
        for (a = 0; a < w->node_count; a++)
        {
            w->indirect_plane->node_pvectors[a] = NULL;
        }
        for (idx = 0; idx < w->edge_count; idx++)
        {
            w->indirect_plane->edge_pvectors[idx] = NULL;
        }

        evaluate_site_likelihood(lhood, m->g, m->preorder, w);

        /*
         * If any site likelihood is exactly zero
         * then we do not need to continue.
         * todo: if the site likelihood interval includes zero
         *       then this could be handled at this point,
         *       for example by immediately requesting higher precision
         */
        if (arb_is_zero(lhood))
        {
            fprintf(stderr, "error: infeasible\n");
            result = -1;
            goto finish;
        }

        /*
         * Evaluate first and second derivatives of site likelihood
         * with respect to each edge rate coefficient.
         * Note that these derivatives are with respect to the likelihood
         * rather than with respect to the log likelihood.
         */
        for (i = 0; i < w->edge_count; i++)
        {
            int update_indirect;

            /* Point to site likelihood vectors. */
            for (a = 0; a < w->node_count; a++)
            {
                w->indirect_plane->node_pvectors[a] = (
                        w->lhood_plane->node_vectors + a);
            }
            for (idx = 0; idx < w->edge_count; idx++)
            {
                w->indirect_plane->edge_pvectors[idx] = (
                        w->lhood_plane->edge_vectors + idx);
            }

            update_indirect = 1;
            evaluate_site_derivatives(lhood_gradient + i,
                    m->g, m->preorder, w, idx_to_a, b_to_idx,
                    i, w->deriv_plane, update_indirect);
            for (j = 0; j <= i; j++)
            {
                update_indirect = 0;
                evaluate_site_derivatives(
                        arb_mat_entry(lhood_hessian, i, j),
                        m->g, m->preorder, w, idx_to_a, b_to_idx,
                        j, w->hess_plane, update_indirect);
            }
        }

        /* define a site-specific coefficient */
        arb_div(x, site_weights+site, site_weight_divisor, prec);
        arb_div(x, x, lhood, prec);

        /*
         * Compute the log likelihood hessian for this site, given:
         *  - likelihood
         *  - likelihood gradient
         *  - likelihood hessian
         *
         * d^2/dxdy log(f(x,y))
         * = (d^2/dxdy f(x,y)) / f(x,y) -
         *   ((d/dx f(x,y)) * (d/dy f(x,y))) / f(x,y)^2
         * = ((d^2/dxdy f(x,y)) -
         *   ((d/dx f(x,y)) * (d/dy f(x,y))) / f(x,y)) / f(x,y)
         *    
         */
        arb_mat_zero(ll_hessian);
        {
            arb_t tmp;
            arb_init(tmp);
            for (i = 0; i < edge_count; i++)
            {
                for (j = 0; j <= i; j++)
                {
                    arb_ptr ll_hess_entry;
                    ll_hess_entry = arb_mat_entry(ll_hessian, i, j);

                    arb_ptr lhood_hess_entry;
                    lhood_hess_entry = arb_mat_entry(lhood_hessian, i, j);

                    arb_mul(tmp, lhood_gradient+i, lhood_gradient+j, prec);
                    arb_div(tmp, tmp, lhood, prec);
                    arb_sub(ll_hess_entry, lhood_hess_entry, tmp, prec);

                    /* x is (site weight) / (site likelihood) */
                    arb_mul(ll_hess_entry, ll_hess_entry, x, prec);
                }
            }
            arb_clear(tmp);
        }

        /* accumulate log likelihood */
        {
            arb_t ll;
            arb_init(ll);
            arb_log(ll, lhood, prec);
            arb_mul(ll, ll, site_weights + site, prec);
            arb_div(ll, ll, site_weight_divisor, prec);
            arb_add(so->ll, so->ll, ll, prec);
            arb_clear(ll);
        }

        /* accumulate gradient of log likelihood */
        for (i = 0; i < edge_count; i++)
        {
            for (state = 0; state < state_count; state++)
            {
                arb_addmul(
                        so->ll_gradient + state,
                        lhood_gradient + state,
                        x, prec);
            }
        }

        /* accumulate hessian of log likelihood */
        arb_mat_add(so->ll_hessian, so->ll_hessian, ll_hessian, prec);
    }

finish:

    likelihood_ws_clear(w);

    arb_clear(site_weight_divisor);
    _arb_vec_clear(site_weights, site_count);

    arb_clear(x);
    arb_clear(lhood);
    _arb_vec_clear(lhood_gradient, edge_count);
    arb_mat_clear(lhood_hessian);
    arb_mat_clear(ll_hessian);

    free(idx_to_a);
    free(b_to_idx);

    return result;
}


typedef struct
{
    int edge_count;
    arb_t ll;
    arb_struct * ll_gradient;
    arb_mat_t ll_hessian;
} second_order_struct;
typedef second_order_struct second_order_t[1];

static void
second_order_init(second_order_t so, int edge_count)
{
    so->edge_count = edge_count;
    arb_init(so->ll);
    so->ll_gradient = _arb_vec_init(edge_count);
    arb_mat_init(so->ll_hessian, edge_count, edge_count);
}

static void
second_order_clear(second_order_t so)
{
    arb_clear(so->ll);
    _arb_vec_clear(so->ll_gradient, so->edge_count);
    arb_mat_clear(so->ll_hessian);
}

static void
second_order_zero(second_order_t so)
{
    arb_zero(so->ll);
    _arb_vec_zero(so->ll_gradient, so->edge_count);
    arb_mat_zero(so->ll, );
}



static json_t *
_hess_agg_site_edge_yn(
        model_and_data_t m,
        column_reduction_t r_site, int *site_selection_count, int *result_out)
{
    json_t * j_out = NULL;
    int result = 0;
    int i, j;
    slong prec;
    int failed;

    int edge_count = m->g->nnz;

    second_order_t so;
    second_order_init(so, edge_count);

    /* repeat with increasing precision until there is no precision failure */
    for (failed=1, prec=4; failed; prec<<=1)
    {

        result = _compute_second_order(
                so, m, r_site, site_selection_count, prec);
        if (result) goto finish;

        /*
         * Check for bounds failure of any entry of the hessian matrix.
         * todo: check in a more fine-grained manner,
         *       to avoid unnecessarily recomputing some hessian entries
         *       at higher precision.
         */
        failed = 0;
        for (i = 0; i < edge_count; i++)
        {
            for (j = 0; j <= i; j++)
            {
                arb_ptr p;
                p = arb_mat_entry(so->ll_hessian, i, j);
                if (!_can_round(p))
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
        int first_edge, second_edge;
        for (first_edge = 0; first_edge < edge_count; first_edge++)
        {
            for (second_edge = 0; second_edge < edge_count; second_edge++)
            {
                arb_ptr p;
                i = m->edge_map->order[first_edge];
                j = m->edge_map->order[second_edge];
                if (j < i)
                {
                    p = arb_mat_entry(so->ll_hessian, i, j);
                }
                else
                {
                    p = arb_mat_entry(so->ll_hessian, j, i);
                }
                d = arf_get_d(arb_midref(p), ARF_RND_NEAR);
                x = json_pack("[i, i, f]", first_edge, second_edge, d);
                json_array_append_new(j_data, x);
            }
        }
        j_out = json_pack("{s:[s, s, s], s:o}",
                "columns", "first_edge", "second_edge", "value",
                "data", j_data);
    }

finish:

    second_order_clear(so, edge_count);

    *result_out = result;
    return j_out;
}





json_t *arbplf_hess_run(void *userdata, json_t *root, int *retcode)
{
    json_t *j_out = NULL;
    json_t *model_and_data = NULL;
    json_t *site_reduction = NULL;
    int site_count = 0;
    int result = 0;
    model_and_data_t m;
    column_reduction_t r_site;
    int i;
    int site;
    int *site_selection_count = NULL;

    model_and_data_init(m);
    column_reduction_init(r_site);

    if (userdata)
    {
        fprintf(stderr, "error: unexpected userdata\n");
        result = -1;
        goto finish;
    }

    /* parse the json input */
    /* Note that site reduction is required, and edge reduction is forbidden */
    {
        json_error_t err;
        size_t flags;
        
        flags = JSON_STRICT;
        result = json_unpack_ex(root, &err, flags,
                "{s:o, s:o}",
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

    /* validate the site reduction section of the json input */
    result = validate_column_reduction(
            r_site, site_count, "site", site_reduction);
    if (result) goto finish;

    if (r_site->agg_mode == AGG_NONE)
    {
        fprintf(stderr, "error: aggregation over sites is required\n");
        result = -1;
        goto finish;
    }

    /* Indicate which site indices are included in the selection. */
    site_selection_count = calloc(site_count, sizeof(int));
    for (i = 0; i < r_site->selection_len; i++)
    {
        site = r_site->selection[i];
        site_selection_count[site]++;
    }

    j_out = _hess_agg_site_edge_yn(m, r_site, site_selection_count, &result);
    if (result) goto finish;

finish:

    *retcode = result;

    free(site_selection_count);
    column_reduction_clear(r_site);
    model_and_data_clear(m);

    flint_cleanup();
    return j_out;
}
