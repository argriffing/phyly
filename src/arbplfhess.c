/*
 * Use arbitrary precision matrix operations to compute functions related
 * to second order approximations of phylogenetic log likelihood functions.
 *
 * These functions include:
 * 1) computing the hessian matrix
 * 2) computing the inverse of the hessian matrix
 * 3) computing the newton delta
 * 4) computing the newton point
 * 5) certifying a local optimum using the interval newton method
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
#include "evaluate_site_lhood.h"
#include "arb_calc.h"
#include "arb_vec_extras.h"
#include "arb_mat_extras.h"
#include "arb_vec_calc.h"
#include "arb_vec_calc_quad.h"
#include "finite_differences.h"
#include "equilibrium.h"
#include "cross_site_ws.h"

#include "parsemodel.h"
#include "parsereduction.h"
#include "runjson.h"
#include "arbplfhess.h"

/*
static void
_print_arb_vec_containment_details(
        const arb_struct *a, const arb_struct *b, slong n)
{
    slong i;
    flint_printf("(e)qual (c)ontains (o)verlaps (n)one:\n");
    for (i = 0; i < n; i++)
    {
        if (arb_equal(a + i, b + i))
            flint_printf("e");
        else if (arb_contains(a + i, b + i))
            flint_printf("c");
        else if (arb_overlaps(a + i, b + i))
            flint_printf("o");
        else
            flint_printf("n");
    }
    flint_printf("\n");
}
*/


/* this struct does not own its vectors */
typedef struct
{
    arb_mat_struct **node_pvectors;
    arb_mat_struct **edge_pvectors;
} indirect_plane_struct;
typedef indirect_plane_struct indirect_plane_t[1];

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


/* second order approximation of a log likelihood */
typedef struct
{
    int edge_count;
    arb_t ll;
    arb_struct * x;
    arb_struct * ll_gradient;
    arb_mat_t ll_hessian;
} so_struct;
typedef so_struct so_t[1];

static void
so_init(so_t so, int edge_count)
{
    so->edge_count = edge_count;
    arb_init(so->ll);
    so->x = _arb_vec_init(edge_count);
    so->ll_gradient = _arb_vec_init(edge_count);
    arb_mat_init(so->ll_hessian, edge_count, edge_count);
}

static void
so_clear(so_t so)
{
    arb_clear(so->ll);
    _arb_vec_clear(so->x, so->edge_count);
    _arb_vec_clear(so->ll_gradient, so->edge_count);
    arb_mat_clear(so->ll_hessian);
}

static void
so_zero(so_t so)
{
    arb_zero(so->ll);
    _arb_vec_zero(so->x, so->edge_count);
    _arb_vec_zero(so->ll_gradient, so->edge_count);
    arb_mat_zero(so->ll_hessian);
}

static void
so_indeterminate(so_t so)
{
    arb_indeterminate(so->ll);
    _arb_vec_indeterminate(so->x, so->edge_count);
    _arb_vec_indeterminate(so->ll_gradient, so->edge_count);
    _arb_mat_indeterminate(so->ll_hessian);
}

static int
so_get_inv_hess(arb_mat_t A, so_t so, slong prec)
{
    int invertible;
    invertible = arb_mat_inv(A, so->ll_hessian, prec);
    if (!invertible)
    {
        _arb_mat_indeterminate(A);
    }
    return invertible;
}

static int
so_hessian_is_negative_definite(const so_t so, slong prec)
{
    int ret;
    slong n;
    arb_mat_t A, L;

    n = arb_mat_nrows(so->ll_hessian);

    arb_mat_init(A, n, n);
    arb_mat_init(L, n, n);

    arb_mat_neg(A, so->ll_hessian);

    ret = arb_mat_cho(L, A, prec);

    arb_mat_clear(A);
    arb_mat_clear(L);

    return ret;
}

static void
so_get_newton_delta(arb_struct * newton_delta, so_t so, slong prec)
{
    int i, n;
    arb_mat_t grad;
    arb_mat_t u;
    int invertible;

    n = so->edge_count;

    /* newton_delta = -u = -inv(hess)*grad */
    arb_mat_init(u, n, 1);
    arb_mat_init(grad, n, 1);
    for (i = 0; i < n; i++)
    {
        arb_set(arb_mat_entry(grad, i, 0), so->ll_gradient + i);
    }

    invertible = arb_mat_solve(u, so->ll_hessian, grad, prec);
    if (!invertible)
    {
        _arb_mat_indeterminate(u);
    }

    for (i = 0; i < n; i++)
    {
        arb_neg(newton_delta + i, arb_mat_entry(u, i, 0));
    }

    arb_mat_clear(grad);
    arb_mat_clear(u);
}

static void
so_get_newton_point(arb_struct * newton_point, so_t so, slong prec)
{
    int n;
    n = so->edge_count;
    so_get_newton_delta(newton_point, so, prec);
    _arb_vec_add(newton_point, so->x, newton_point, n, prec);
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


typedef struct
{
    plane_t base_plane;
    plane_t lhood_plane;
    plane_t deriv_plane;
    plane_t hess_plane;

    indirect_plane_t indirect_plane;
} likelihood_ws_struct;
typedef likelihood_ws_struct likelihood_ws_t[1];

static void
likelihood_ws_init(likelihood_ws_t w, const model_and_data_t m)
{
    slong node_count = model_and_data_node_count(m);
    slong edge_count = model_and_data_edge_count(m);
    slong state_count = model_and_data_state_count(m);

    plane_init(w->base_plane, node_count, edge_count, state_count);
    plane_init(w->lhood_plane, node_count, edge_count, state_count);
    plane_init(w->deriv_plane, node_count, edge_count, state_count);
    plane_init(w->hess_plane, node_count, edge_count, state_count);

    indirect_plane_init(w->indirect_plane, node_count, edge_count);
}

static void
likelihood_ws_clear(likelihood_ws_t w)
{
    plane_clear(w->base_plane);
    plane_clear(w->lhood_plane);
    plane_clear(w->deriv_plane);
    plane_clear(w->hess_plane);

    indirect_plane_clear(w->indirect_plane);
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
        const arb_mat_struct *rate_matrix,
        const arb_mat_struct *transition_matrices,
        const root_prior_t r, const arb_struct *equilibrium,
        likelihood_ws_t w, int *idx_to_a, int *b_to_idx,
        int deriv_idx, plane_t plane, int update_indirect, slong prec)
{
    int a, b, idx, curr_idx;
    int start, stop;
    arb_mat_struct * nmat;
    arb_mat_struct * nmatb;
    arb_mat_struct * emat;
    arb_mat_struct * emati;
    const arb_mat_struct * rmat;
    const arb_mat_struct * tmat;
    indirect_plane_struct * indirect;

    rmat = rate_matrix;
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

        /* debug */
        /*
        flint_printf("a=%d nmat before:\n", a);
        arb_mat_printd(nmat, 15); flint_printf("\n");
        */

        for (idx = start; idx < stop; idx++)
        {
            if (idx == deriv_idx)
            {
                emati = indirect->edge_pvectors[idx];
                emat = plane->edge_vectors + idx;
                _arb_mat_mul_rate_matrix(emat, rmat, emati, prec);
                arb_mat_mul_entrywise(nmat, nmat, emat, prec);
                if (update_indirect)
                {
                    indirect->edge_pvectors[idx] = emat;
                    indirect->node_pvectors[a] = nmat;
                }
            }
            else if (idx == curr_idx)
            {
                b = g->indices[idx];
                tmat = transition_matrices + idx;
                nmatb = plane->node_vectors + b;
                emat = plane->edge_vectors + idx;
                _arb_mat_mul_stochastic(emat, tmat, nmatb, prec);
                arb_mat_mul_entrywise(nmat, nmat, emat, prec);
                if (update_indirect)
                {
                    indirect->edge_pvectors[idx] = emat;
                    indirect->node_pvectors[a] = nmat;
                }
            }
            else
            {
                emati = indirect->edge_pvectors[idx];
                arb_mat_mul_entrywise(nmat, nmat, emati, prec);
                if (update_indirect)
                {
                    indirect->node_pvectors[a] = nmat;
                }
            }
        }

        /* debug */
        /*
        flint_printf("a=%d nmat after:\n", a);
        arb_mat_printd(nmat, 15); flint_printf("\n");
        */

        /* move back towards the root of the tree */
        curr_idx = b_to_idx[a];
    }


    /* Report the sum of state entries associated with the root. */
    nmat = plane->node_vectors + preorder[0];
    root_prior_expectation(derivative, r, nmat, equilibrium, prec);

    /* debug */
    /*
    flint_printf("final nmat:\n");
    arb_mat_printd(nmat, 15); flint_printf("\n");
    flint_printf("final derivative:\n");
    arb_printd(derivative, 15); flint_printf("\n");
    */
}


/*
 * This is a helper function for _recompute_second_order().
 *
 * d^2/dxdy log(f(x,y))
 * = (d^2/dxdy f(x,y)) / f(x,y) -
 *   ((d/dx f(x,y)) * (d/dy f(x,y))) / f(x,y)^2
 * = ((d^2/dxdy f(x,y)) -
 *   ((d/dx f(x,y)) * (d/dy f(x,y))) / f(x,y)) / f(x,y)
 */
static void
_lhood_hess_to_ll_hess(arb_mat_t ll_hessian,
        arb_t lhood, arb_struct *lhood_gradient, arb_mat_t lhood_hessian,
        const int *pre_to_idx, slong prec)
{
    int i, j;
    arb_t tmp;
    slong edge_count = arb_mat_nrows(lhood_hessian);

    /* debug */
    /*
    flint_printf("lhood hessian:\n");
    arb_mat_printd(lhood_hessian, 15); flint_printf("\n");
    */

    arb_init(tmp);
    arb_mat_zero(ll_hessian);
    for (i = 0; i < edge_count; i++)
    {
        for (j = 0; j <= i; j++)
        {
            int a = pre_to_idx[i];
            int b = pre_to_idx[j];
            arb_ptr ll_hess_entry = arb_mat_entry(ll_hessian, a, b);
            arb_ptr lhood_hess_entry = arb_mat_entry(lhood_hessian, a, b);

            arb_mul(tmp, lhood_gradient+a, lhood_gradient+b, prec);
            arb_div(tmp, tmp, lhood, prec);
            arb_sub(ll_hess_entry, lhood_hess_entry, tmp, prec);
            arb_div(ll_hess_entry, ll_hess_entry, lhood, prec);
        }
    }

    /* debug */
    /*
    flint_printf("ll hessian:\n");
    arb_mat_printd(ll_hessian, 15); flint_printf("\n");
    */

    arb_clear(tmp);
}


/*
 * Optionally allow custom edge rate coefficients.
 * {y, g, h} indicate the function evaluation and its gradient and hessian.
 */
static int
_recompute_second_order(so_t so,
        model_and_data_t m, column_reduction_t r_site,
        const arb_struct *edge_rates,
        int req_y, int req_g, int req_h,
        slong prec)
{
    cross_site_ws_t csw;
    likelihood_ws_t w;
    int result = 0;
    int i, j, idx, site, a;

    int *site_selection_count = NULL;
    int *pre_to_idx = NULL;

    slong site_count = model_and_data_site_count(m);
    slong edge_count = model_and_data_edge_count(m);
    slong node_count = model_and_data_node_count(m);
    slong rate_category_count = model_and_data_rate_category_count(m);

    arb_t cat_rate_squared, rate;

    arb_t tmp;
    arb_mat_t ll_hessian;

    arb_struct * site_weights;
    arb_t site_weight_divisor;

    arb_init(cat_rate_squared);
    arb_init(rate);

    /* category-specific lhood-related values */
    arb_t cat_lhood;
    arb_struct * cat_lhood_gradient;
    arb_mat_t cat_lhood_hessian;

    /* site-specific lhood-related values, accumulated across categories */
    arb_t site_lhood;
    arb_struct * site_lhood_gradient;
    arb_mat_t site_lhood_hessian;

    /* init category-specific lhood-related values */
    arb_init(cat_lhood);
    cat_lhood_gradient = _arb_vec_init(edge_count);
    arb_mat_init(cat_lhood_hessian, edge_count, edge_count);

    /* init site-specific lhood-related values */
    arb_init(site_lhood);
    site_lhood_gradient = _arb_vec_init(edge_count);
    arb_mat_init(site_lhood_hessian, edge_count, edge_count);

    arb_init(tmp);
    arb_mat_init(ll_hessian, edge_count, edge_count);

    /* Initialize the outputs to zero, before accumulating across sites. */
    so_zero(so);

    arb_init(site_weight_divisor);
    site_weights = _arb_vec_init(site_count);

    /* Indicate which site indices are included in the selection. */
    site_selection_count = calloc(site_count, sizeof(int));
    for (i = 0; i < r_site->selection_len; i++)
    {
        site = r_site->selection[i];
        site_selection_count[site]++;
    }

    /* Get maps for navigating towards the root of the tree. */
    pre_to_idx = malloc(edge_count * sizeof(int));
    _csr_graph_get_preorder_edges(pre_to_idx, m->g, m->navigation->preorder);

    /* define site aggregation weights */
    result = get_site_agg_weights(
            site_weight_divisor, site_weights, site_count, r_site, prec);
    if (result) goto finish;

    cross_site_ws_init(csw, m);
    cross_site_ws_update_with_edge_rates(csw, m, edge_rates, prec);
    likelihood_ws_init(w, m);
    _arb_vec_set(so->x, csw->edge_rates, edge_count);

    for (site = 0; site < site_count; site++)
    {
        slong cat;

        /* skip sites that are not selected */
        if (!site_selection_count[site]) continue;

        /*
         * Initialize base plane nodes for this site
         * according to prior state distributions and data.
         * Edges remain unused.
         */
        pmat_update_base_node_vectors(w->base_plane->node_vectors, m->p, site);

        /* zero cross-category lhood accumulators */
        arb_zero(site_lhood);
        _arb_vec_zero(site_lhood_gradient, edge_count);
        arb_mat_zero(site_lhood_hessian);

        for (cat = 0; cat < rate_category_count; cat++)
        {
            const arb_struct * cat_rate = csw->rate_mix_rates + cat;
            const arb_struct * prior_prob = csw->rate_mix_prior + cat;
            arb_mat_struct *tmat_base;
            tmat_base = cross_site_ws_transition_matrix(csw, cat, 0);

            /* Reset pointers in the virtual plane. */
            for (a = 0; a < node_count; a++)
            {
                w->indirect_plane->node_pvectors[a] = NULL;
            }
            for (idx = 0; idx < edge_count; idx++)
            {
                w->indirect_plane->edge_pvectors[idx] = NULL;
            }

            evaluate_site_lhood(cat_lhood,
                    w->lhood_plane->node_vectors,
                    w->lhood_plane->edge_vectors,
                    w->base_plane->node_vectors,
                    m->root_prior, csw->equilibrium,
                    tmat_base,
                    m->g, m->navigation->preorder, node_count, prec);

            /* compute category likelihood */
            arb_mul(cat_lhood, cat_lhood, prior_prob, prec);

            if (arb_is_zero(cat_lhood))
            {
                if (arb_is_zero(cat_rate))
                {
                    /*
                     * If the category rate is zero and the site is infeasible,
                     * then changing the edge rate scaling factors will not
                     * rescue the feasibility, so the derivatives should
                     * all be zero.
                     */
                    continue;
                }
                else
                {
                    /* todo: allow some rate categories to be infeasible */
                    if (arb_is_zero(cat_lhood))
                    {
                        fprintf(stderr, "error: infeasible\n");
                        result = -1;
                        goto finish;
                    }

                    if (arb_contains_zero(cat_lhood))
                    {
                        so_indeterminate(so);
                        goto finish;
                    }
                }
            }

            /*
             * Evaluate first and second derivatives of site likelihood
             * with respect to each edge rate coefficient.
             * Note that these derivatives are with respect to the likelihood
             * rather than with respect to the log likelihood.
             */
            if (req_g || req_h)
            {
                for (i = 0; i < edge_count; i++)
                {
                    int update_indirect;
                    int idx_i;

                    idx_i = pre_to_idx[i];

                    /* Point to site likelihood vectors. */
                    for (a = 0; a < node_count; a++)
                    {
                        w->indirect_plane->node_pvectors[a] = (
                                w->lhood_plane->node_vectors + a);
                    }
                    for (idx = 0; idx < edge_count; idx++)
                    {
                        w->indirect_plane->edge_pvectors[idx] = (
                                w->lhood_plane->edge_vectors + idx);
                    }

                    update_indirect = 1;
                    evaluate_site_derivatives(cat_lhood_gradient + idx_i,
                            m->g, m->navigation->preorder,
                            csw->rate_matrix, tmat_base,
                            m->root_prior, csw->equilibrium,
                            w, m->navigation->idx_to_a, m->navigation->b_to_idx,
                            idx_i, w->deriv_plane, update_indirect, prec);

                    if (req_h)
                    {
                        for (j = 0; j <= i; j++)
                        {
                            int idx_j;
                            idx_j = pre_to_idx[j];
                            update_indirect = 0;
                            evaluate_site_derivatives(
                                    arb_mat_entry(
                                        cat_lhood_hessian, idx_i, idx_j),
                                    m->g, m->navigation->preorder,
                                    csw->rate_matrix, tmat_base,
                                    m->root_prior, csw->equilibrium,
                                    w, m->navigation->idx_to_a, m->navigation->b_to_idx,
                                    idx_j, w->hess_plane, update_indirect, prec);
                        }
                    }
                }
            }

            /* Accumulate the lhood data. */
            arb_add(site_lhood, site_lhood, cat_lhood, prec);
            arb_mul(cat_rate_squared, cat_rate, cat_rate, prec);
            for (i = 0; i < edge_count; i++)
            {
                int a = pre_to_idx[i];
                arb_struct *dsite = site_lhood_gradient + a;
                arb_struct *dcat = cat_lhood_gradient + a;

                /* accumulate gradient */
                arb_mul(tmp, cat_rate, prior_prob, prec);
                arb_addmul(dsite, dcat, tmp, prec);

                /* accumulate hessian */
                arb_mul(tmp, tmp, cat_rate, prec);
                for (j = 0; j <= i; j++)
                {
                    int b = pre_to_idx[j];
                    arb_struct *hsite = arb_mat_entry(site_lhood_hessian, a, b);
                    arb_struct *hcat = arb_mat_entry(cat_lhood_hessian, a, b);
                    arb_addmul(hsite, hcat, tmp, prec);
                }
            }
        }

        /*
         * Now that site likelihood data has been computed across categories,
         * accumulate the data across sites in log space.
         */

        /*
         * Compute the hessian of log likelihood for this site, given:
         *  - likelihood
         *  - likelihood gradient
         *  - likelihood hessian
         *
         * accumulate the hessian of log likelihood
         */
        if (req_h)
        {
            _lhood_hess_to_ll_hess(ll_hessian,
                    site_lhood, site_lhood_gradient, site_lhood_hessian,
                    pre_to_idx, prec);
            arb_mat_scalar_mul_arb(ll_hessian,
                    ll_hessian, site_weights+site, prec);
            arb_mat_scalar_div_arb(ll_hessian,
                    ll_hessian, site_weight_divisor, prec);
            arb_mat_add(so->ll_hessian, so->ll_hessian, ll_hessian, prec);
        }

        /* accumulate log likelihood */
        {
            arb_log(tmp, site_lhood, prec);
            arb_div(tmp, tmp, site_weight_divisor, prec);
            arb_addmul(so->ll, site_weights + site, tmp, prec);
        }

        /* accumulate gradient of log likelihood */
        if (req_g)
        {
            arb_div(tmp, site_weights + site, site_lhood, prec);
            arb_div(tmp, tmp, site_weight_divisor, prec);
            for (i = 0; i < edge_count; i++)
            {
                arb_addmul(so->ll_gradient + i,
                        site_lhood_gradient + i, tmp, prec);
            }
        }
    }

    /* complete the hessian according to symmetry */
    if (req_h)
    {
        for (i = 0; i < edge_count; i++)
        {
            for (j = 0; j < i; j++)
            {
                int a = pre_to_idx[i];
                int b = pre_to_idx[j];
                arb_set(arb_mat_entry(so->ll_hessian, b, a),
                        arb_mat_entry(so->ll_hessian, a, b));
            }
        }
    }

finish:

    cross_site_ws_clear(csw);
    likelihood_ws_clear(w);

    arb_clear(site_weight_divisor);
    _arb_vec_clear(site_weights, site_count);

    arb_clear(cat_rate_squared);
    arb_clear(rate);

    /* clear category-specific lhood-related values */
    arb_clear(cat_lhood);
    _arb_vec_clear(cat_lhood_gradient, edge_count);
    arb_mat_clear(cat_lhood_hessian);

    /* clear site-specific lhood-related values */
    arb_clear(site_lhood);
    _arb_vec_clear(site_lhood_gradient, edge_count);
    arb_mat_clear(site_lhood_hessian);

    arb_clear(tmp);
    arb_mat_clear(ll_hessian);

    free(pre_to_idx);
    free(site_selection_count);

    return result;
}


/*
 * Second order log likelihood computations aggregated over sites.
 * The calculations use a specified working precision.
 */
static int
_compute_second_order(so_t so,
        model_and_data_t m, column_reduction_t r_site, slong prec)
{
    arb_struct *edge_rates;
    edge_rates = NULL;
    return _recompute_second_order(so, m, r_site, edge_rates, 1, 1, 1, prec);
}




/* this struct does not 'own' its data */
typedef struct
{
    model_and_data_struct *m;
    column_reduction_struct *r_site;
} _objective_param_struct;

static void
_objective_param_init(_objective_param_struct *s,
        model_and_data_struct *m,
        column_reduction_struct *r_site)
{
    s->m = m;
    s->r_site = r_site;
}

static void
_objective_param_clear(_objective_param_struct *s)
{
    s->m = NULL;
    s->r_site = NULL;
}


/*
 * For finite differences calculation of gradient and hessian.
 * This follows the 'multivariate_function_t' interface in 
 * the finite_differences module.
 */
void _compute_ll(arb_t ll,
        const arb_struct *edge_rates, void *param,
        slong n, slong prec)
{
    slong i;
    cross_site_ws_t csw;
    likelihood_ws_t w;
    int idx, site, a;
    int *site_selection_count;
    int result = 0;

    arb_t x;
    arb_t site_lhood;
    arb_t site_ll;

    _objective_param_struct *p = param;
    const model_and_data_struct *m = p->m;
    slong site_count = model_and_data_site_count(m);
    slong edge_count = model_and_data_edge_count(m);
    slong node_count = model_and_data_node_count(m);

    site_selection_count = NULL;

    if (edge_count != n)
    {
        flint_printf("internal error: edge count inconsistency\n");
        abort();
    }

    arb_init(x);
    arb_init(site_lhood);
    arb_init(site_ll);

    arb_struct * site_weights;
    arb_t site_weight_divisor;

    /* initialize output log likelihood to zero for accumulation */
    arb_zero(ll);

    arb_init(site_weight_divisor);
    site_weights = _arb_vec_init(site_count);

    /* Indicate which site indices are included in the selection. */
    site_selection_count = calloc(site_count, sizeof(int));
    for (i = 0; i < p->r_site->selection_len; i++)
    {
        site = p->r_site->selection[i];
        site_selection_count[site]++;
    }

    /* define site aggregation weights */
    result = get_site_agg_weights(
            site_weight_divisor, site_weights, site_count, p->r_site, prec);
    if (result) goto finish;

    cross_site_ws_init(csw, m);
    cross_site_ws_update_with_edge_rates(csw, m, edge_rates, prec);
    likelihood_ws_init(w, m);

    for (site = 0; site < site_count; site++)
    {
        slong cat;
        const arb_mat_struct *tmat_base;

        /* skip sites that are not selected */
        if (!site_selection_count[site]) continue;

        /* todo: allow multiple rate categories */
        cat = 0;
        tmat_base = cross_site_ws_transition_matrix(csw, cat, 0);

        /*
         * Initialize base plane nodes for this site
         * according to prior state distributions and data.
         * Edges remain unused.
         */
        pmat_update_base_node_vectors(w->base_plane->node_vectors, m->p, site);

        /* Reset pointers in the virtual plane. */
        for (a = 0; a < node_count; a++)
        {
            w->indirect_plane->node_pvectors[a] = NULL;
        }
        for (idx = 0; idx < edge_count; idx++)
        {
            w->indirect_plane->edge_pvectors[idx] = NULL;
        }

        evaluate_site_lhood(site_lhood,
                w->lhood_plane->node_vectors,
                w->lhood_plane->edge_vectors,
                w->base_plane->node_vectors,
                m->root_prior, csw->equilibrium,
                tmat_base,
                m->g, m->navigation->preorder, node_count, prec);

        /*
         * If any site likelihood is exactly zero
         * then we do not need to continue.
         */
        if (arb_is_zero(site_lhood))
        {
            fprintf(stderr, "error: infeasible\n");
            result = -1;
            goto finish;
        }

        /*
         * If the site likelihood interval includes zero
         * then this could be handled at this point,
         * for example by immediately requesting higher precision.
         */
        if (arb_contains_zero(site_lhood))
        {
            arb_indeterminate(ll);
            goto finish;
        }

        arb_log(site_ll, site_lhood, prec);
        arb_addmul(ll, site_ll, site_weights + site, prec);
    }

    arb_div(ll, ll, site_weight_divisor, prec);

finish:

    cross_site_ws_clear(csw);
    likelihood_ws_clear(w);

    arb_clear(site_weight_divisor);
    _arb_vec_clear(site_weights, site_count);

    arb_clear(x);
    arb_clear(site_lhood);
    arb_clear(site_ll);

    free(site_selection_count);
}


static int
_compute_second_order_finite_differences(so_t so,
        model_and_data_t m, column_reduction_t r_site, slong prec)
{
    _objective_param_struct p[1];
    arb_t delta;
    slong edge_count;
    arb_struct *edge_rates;
    gradient_param_t g;

    edge_count = m->g->nnz;
    edge_rates = _arb_vec_init(edge_count);

    /* Extract initial edge rates. */
    {
        slong i;
        slong idx;
        double tmpd;
        for (i = 0; i < edge_count; i++)
        {
            idx = m->edge_map->order[i];
            flint_printf("debug: i=%wd idx=%wd\n", i, idx);
            tmpd = m->edge_rate_coefficients[i];
            arb_set_d(edge_rates + idx, tmpd);
        }
    }

    arb_init(delta);
    _arb_set_si_2exp_si(delta, 1, -(2 + (prec >> 2)));

    /* parameters for log likelihood calculation */
    _objective_param_init(p, m, r_site);

    /* parameters for gradient calculation */
    gradient_param_init(g, _compute_ll, p, delta);

    finite_differences_gradient(
            so->ll_gradient, edge_rates,
            g, edge_count, edge_count, prec);

    finite_differences_hessian(
            so->ll_hessian, _compute_ll, p,
            edge_rates, edge_count, delta, prec);

    arb_clear(delta);
    _objective_param_clear(p);
    _arb_vec_clear(edge_rates, edge_count);
    gradient_param_clear(g);

    return 0;
}


/*
 * This is the function whose zeros (roots) are of interest.
 * It follows the interface of arb_vec_calc_func_t.
 */
static int
_objective(arb_struct *vec_out, arb_mat_struct *jac_out,
      const arb_struct *inp, void *param, slong n, slong prec)
{
    _objective_param_struct * s = param;
    int result;
    so_t so;

    so_init(so, n);

    /*
     * Evaluate the function and its jacobian.
     * the {1, 1, 1} args indicate that all calculations are requested,
     * as opposed to skipping higher order Taylor terms.
     */
    result = _recompute_second_order(so, s->m, s->r_site, inp, 1, 1, 1, prec);
    if (result) abort();

    _arb_vec_set(vec_out, so->ll_gradient, n);
    arb_mat_set(jac_out, so->ll_hessian);

    so_clear(so);

    return 0;
}

/*
 * This is the function whose local minimum is of interest.
 * It follows the interface of arb_vec_calc_f_t.
 * The {y, g, h} output arguments correspond to the negative log likelihood
 * and its gradient and hessian respectively. If non-NULL, they will
 * hold the output of the corresponding functions.
 * This is done 'lazily' so for example the hessian will not be
 * computed if only the negative log likelihood and gradient are requested.
 */
static int
_minimization_objective(
        arb_struct *y, arb_struct *g, arb_mat_struct *h,
        const arb_struct *x, void *param, slong n, slong prec)
{
    /* ad-hoc bounds check */
    if (_arb_vec_is_nonnegative(x, n))
    {
        _objective_param_struct * s = param;
        int result;
        so_t so;

        so_init(so, n);

        result = _recompute_second_order(
                so, s->m, s->r_site, x,
                (y != NULL), (g != NULL), (h != NULL), prec);
        if (result) abort();

        if (y)
            arb_neg(y, so->ll);

        if (g)
            _arb_vec_neg(g, so->ll_gradient, n);

        if (h)
            arb_mat_neg(h, so->ll_hessian);

        so_clear(so);

        return 0;
    }
    else
    {
        if (y)
            arb_indeterminate(y);

        if (g)
            _arb_vec_indeterminate(g, n);

        if (h)
            _arb_mat_indeterminate(h);

        return -1;
    }
}



/*
 * Parse model and data and site reduction,
 * subject to requirements that are common to all second order queries.
 */
static int
_parse_second_order(model_and_data_t m, column_reduction_t r, json_t *root)
{
    json_t *model_and_data = NULL;
    json_t *site_reduction = NULL;
    int site_count = 0;
    int result = 0;
    json_error_t err;
    size_t flags;
    flags = JSON_STRICT;

    model_and_data_init(m);
    column_reduction_init(r);

    /* parse the json input */
    /* Note that site reduction is required, and edge reduction is forbidden */
    result = json_unpack_ex(root, &err, flags,
            "{s:o, s:o}",
            "model_and_data", &model_and_data,
            "site_reduction", &site_reduction
            );
    if (result)
    {
        fprintf(stderr, "error: on line %d: %s\n", err.line, err.text);
        return result;
    }

    /* validate the model and data section of the json input */
    result = validate_model_and_data(m, model_and_data);
    if (result) return result;

    site_count = pmat_nsites(m->p);

    /* validate the site reduction section of the json input */
    result = validate_column_reduction(r, site_count, "site", site_reduction);
    if (result) return result;

    if (r->agg_mode == AGG_NONE)
    {
        fprintf(stderr, "error: aggregation over sites is required\n");
        return -1;
    }

    return 0;
}

json_t *
inv_hess_query(model_and_data_t m, column_reduction_t r_site, int *result_out)
{
    json_t * j_out = NULL;
    int result = 0;
    int i, j;
    slong prec;
    arb_mat_t inv_ll_hessian;
    so_t so;
    int edge_count;

    edge_count = m->g->nnz;
    so_init(so, edge_count);
    arb_mat_init(inv_ll_hessian, edge_count, edge_count);

    /* repeat with increasing precision until there is no precision failure */
    int success = 0;
    for (prec=4; !success; prec <<= 1)
    {
        int invertible;
        result = _compute_second_order(so, m, r_site, prec);
        if (result) goto finish;

        invertible = so_get_inv_hess(inv_ll_hessian, so, prec);
        if (invertible)
        {
            success = _arb_mat_can_round(inv_ll_hessian);
        }
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
                    p = arb_mat_entry(inv_ll_hessian, i, j);
                }
                else
                {
                    p = arb_mat_entry(inv_ll_hessian, j, i);
                }
                d = _arb_get_d(p);
                x = json_pack("[i, i, f]", first_edge, second_edge, d);
                json_array_append_new(j_data, x);
            }
        }
        j_out = json_pack("{s:[s, s, s], s:o}",
                "columns", "first_edge", "second_edge", "value",
                "data", j_data);
    }

finish:

    so_clear(so);
    arb_mat_clear(inv_ll_hessian);

    *result_out = result;
    return j_out;
}


json_t *
hess_query(model_and_data_t m, column_reduction_t r_site, int *result_out)
{
    json_t * j_out = NULL;
    int result = 0;
    int i, j;
    slong prec;

    int edge_count = m->g->nnz;

    so_t so;
    so_init(so, edge_count);

    /* repeat with increasing precision until there is no precision failure */
    int success = 0;
    for (prec=4; !success; prec <<= 1)
    {
        result = _compute_second_order(
        /* result = _compute_second_order_finite_differences( */
                so, m, r_site, prec);
        if (result) goto finish;

        success = _arb_mat_can_round(so->ll_hessian);
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
                d = _arb_get_d(p);
                /* flint_printf("value %lf signbit %d\n", d, signbit(d)); */
                x = json_pack("[i, i, f]", first_edge, second_edge, d);
                json_array_append_new(j_data, x);
            }
        }
        j_out = json_pack("{s:[s, s, s], s:o}",
                "columns", "first_edge", "second_edge", "value",
                "data", j_data);
    }

finish:

    so_clear(so);

    *result_out = result;
    return j_out;
}


json_t *
newton_delta_query(
        model_and_data_t m, column_reduction_t r_site, int *result_out)
{
    json_t * j_out = NULL;
    int result = 0;
    slong prec;
    int i, edge_count;
    so_t so;
    arb_struct *newton_delta;
    
    edge_count = m->g->nnz;
    so_init(so, edge_count);
    newton_delta = _arb_vec_init(edge_count);

    /* repeat with increasing precision until there is no precision failure */
    int success = 0;
    for (prec=4; !success; prec <<= 1)
    {
        result = _compute_second_order(so, m, r_site, prec);
        if (result) goto finish;

        so_get_newton_delta(newton_delta, so, prec);

        success = _arb_vec_can_round(newton_delta, edge_count);
    }

    /* build the json output */
    {
        double d;
        int idx;
        json_t *j_data, *x;
        j_data = json_array();
        for (i = 0; i < edge_count; i++)
        {
            idx = m->edge_map->order[i];
            d = _arb_get_d(newton_delta + idx);
            x = json_pack("[i, f]", i, d);
            json_array_append_new(j_data, x);
        }
        j_out = json_pack("{s:[s, s], s:o}",
                "columns", "edge", "value",
                "data", j_data);
    }

finish:

    so_clear(so);
    _arb_vec_clear(newton_delta, edge_count);

    *result_out = result;
    return j_out;
}


json_t *
newton_point_query(
        model_and_data_t m, column_reduction_t r_site, int *result_out)
{
    json_t * j_out = NULL;
    int result = 0;
    slong prec;
    int i, edge_count;
    so_t so;
    arb_struct *newton_point;
    
    edge_count = m->g->nnz;
    so_init(so, edge_count);
    newton_point = _arb_vec_init(edge_count);

    /* repeat with increasing precision until there is no precision failure */
    int success = 0;
    for (prec=4; !success; prec <<= 1)
    {
        result = _compute_second_order(so, m, r_site, prec);
        if (result) goto finish;

        so_get_newton_point(newton_point, so, prec);

        success = _arb_vec_can_round(newton_point, edge_count);
    }

    /* build the json output */
    {
        double d;
        int idx;
        json_t *j_data, *x;
        j_data = json_array();
        for (i = 0; i < edge_count; i++)
        {
            idx = m->edge_map->order[i];
            d = _arb_get_d(newton_point + idx);
            x = json_pack("[i, f]", i, d);
            json_array_append_new(j_data, x);
        }
        j_out = json_pack("{s:[s, s], s:o}",
                "columns", "edge", "value",
                "data", j_data);
    }

finish:

    so_clear(so);
    _arb_vec_clear(newton_point, edge_count);

    *result_out = result;
    return j_out;
}

json_t *
newton_refine_query(
        model_and_data_t m, column_reduction_t r_site, int *result_out)
{
    json_t * j_out = NULL;
    int result = 0;
    int success;
    slong prec;
    int i, edge_count;
    slong precmax;
    arb_struct *x, *x_initial, *x_preliminary, *x_out;

    /* arb_calc_verbose = 1; */

    edge_count = m->g->nnz;

    x = _arb_vec_init(edge_count);
    x_initial = _arb_vec_init(edge_count);
    x_preliminary = _arb_vec_init(edge_count);
    x_out = _arb_vec_init(edge_count);

    /* Extract initial edge rates to x_initial. */
    {
        slong edge_count;
        slong idx;
        double tmpd;
        edge_count = m->g->nnz;
        for (i = 0; i < edge_count; i++)
        {
            idx = m->edge_map->order[i];
            tmpd = m->edge_rate_coefficients[i];
            arb_set_d(x_initial + idx, tmpd);
        }
    }

    success = 0;
    precmax = 10000;
    for (prec = 32; prec < precmax; prec <<= 1)
    {
        if (arb_calc_verbose)
        {
            flint_printf("preliminary search precision: %wd\n", prec);
        }
        _objective_param_struct s[1];
        myquad_t q_opt, q_initial;

        _objective_param_init(s, m, r_site);

        quad_init(
                q_initial, _minimization_objective,
                x_initial, s, edge_count, prec);
        quad_init_set(q_opt, q_initial);

        {
            arb_t r, rmax;
            slong maxiter;

            maxiter = 10000;

            /*
             * Initial trust radius.
             * Set this to half the minimum initial value.
             */
            {
                arf_t m;
                arf_init(m);
                arf_set(m, arb_midref(q_initial->x + 0));
                for (i = 1; i < edge_count; i++)
                {
                    arf_min(m, m, arb_midref(q_initial->x + i));
                }
                arb_init(r);
                arb_set_arf(r, m);
                arb_mul_2exp_si(r, r, -1);
                arf_clear(m);
            }

            /* max trust radius */
            arb_init(rmax);
            arb_set_d(rmax, 10.0);

            _minimize_dogleg(q_opt, q_initial, r, rmax, maxiter);

            arb_clear(r);
            arb_clear(rmax);
        }

        /* Set the preliminary vector. */
        _arb_vec_set(x_preliminary, q_opt->x, edge_count);

        /*
         * Check if the preliminary solution has reasonably small
         * relative error. Note that at this point,
         * the true solution is not guaranteed to be contained
         * within the preliminary solution vector.
         */
        if (_arb_vec_min_rel_accuracy_bits(x_preliminary, edge_count) > 53)
        {
            success = 1;
        }
        else if (_arb_vec_min_rel_accuracy_bits(x_preliminary, edge_count) > 2)
        {
            /* 
             * If the unsuccessful solution has some amount
             * of significant bits, then set the initial point
             * to the midpoint of the unsuccessful solution.
             */
            _arb_vec_set(x_initial, x_preliminary, edge_count);
            for (i = 0; i < edge_count; i++)
            {
                mag_zero(arb_radref(x_initial + i));
            }
            if (arb_calc_verbose)
            {
                flint_printf("debug: reusing imprecise solution\n");
            }
        }
        else
        {
            if (arb_calc_verbose)
            {
                flint_printf("debug: restarting from scratch\n");
            }
        }

        quad_clear(q_opt);
        quad_clear(q_initial);
        _objective_param_clear(s);

        if (success)
        {
            /* break so that prec is not bumped */
            break;
        }
    }

    if (arb_calc_verbose)
    {
        flint_printf("debug: preliminary solution:\n");
        _arb_vec_printd(x_preliminary, edge_count, 15);
    }

    /*
     * Keep increasing precision until we rule out the possibility
     * of an optimum near the guess provided by the user,
     * or until we find a certified optimum.
     * If the input values are garbage,
     * this loop will keep trying increasing precision without bounds.
     * A more sophisticated search would try subdivision or
     * various branch-and-bound methods, but this is approaching
     * global optimization rather than certification of a local optimum.
     */
    success = 0;
    for (prec = prec; prec < precmax; prec <<= 1)
    {
        int refinement_result;
        slong log2_rtol;

        log2_rtol = -53;

        if (arb_calc_verbose)
            flint_fprintf(stderr, "debug: interval newton prec=%wd\n", prec);

        _arb_vec_set(x, x_preliminary, edge_count);

        /*
         * Inflate the wide intervals by a fixed proportion
         * of the rate coefficient, without regard for the level of precision.
         */
        for (i = 0; i < edge_count; i++)
        {
            mag_t rtmp;

            mag_init(rtmp);
            arb_get_mag(rtmp, x + i);
            mag_mul_2exp_si(rtmp, rtmp, log2_rtol);
            mag_add(arb_radref(x + i), arb_radref(x + i), rtmp);

            mag_clear(rtmp);
        }

        if (arb_calc_verbose)
            _arb_vec_printd(x, edge_count, 15);

        {
            _objective_param_struct s[1];
            _objective_param_init(s, m, r_site);

            /* refinement_result = _arb_vec_calc_refine_root_krawczyk( */
            refinement_result = _arb_vec_calc_refine_root_newton(
                    x_out, _objective, s, x,
                    edge_count, 0, prec);

            _objective_param_clear(s);
        }

        if (arb_calc_verbose)
            flint_printf("debug: refinement_result=%d\n", refinement_result);

        if (refinement_result == -1)
        {
            if (arb_calc_verbose)
                flint_printf("debug: root is excluded\n");
            result = -1;
            goto finish;
        }
        else if (refinement_result == 0)
        {
            if (arb_calc_verbose)
                flint_printf("debug: newton refinement does not "
                             "contract the interval\n");
            result = -1;
        }
        else if (refinement_result == 1)
        {
            success = 1;
            result = 0;
            break;
        }
        else
        {
            flint_printf("internal error: newton refinement result %d\n",
                    refinement_result);
            abort();
        }
    }
    if (prec >= precmax)
    {
        flint_printf("error: newton interval method exceeded precmax\n");
        abort();
    }

    /* require that the hessian is negative definite */
    {
        int negative_definite;
        so_t so;

        so_init(so, edge_count);
        _compute_second_order(so, m, r_site, prec);

        negative_definite = so_hessian_is_negative_definite(so, prec);

        so_clear(so);

        if (!negative_definite)
        {
            fprintf(stderr, "newton refinement fail: log likelihood shape\n");
            result = -1;
            goto finish;
        }
    }

    /* build the json output */
    {
        double d;
        int idx;
        json_t *j_data, *j_x;
        j_data = json_array();
        for (i = 0; i < edge_count; i++)
        {
            idx = m->edge_map->order[i];
            d = _arb_get_d(x_out + idx);
            j_x = json_pack("[i, f]", i, d);
            json_array_append_new(j_data, j_x);
        }
        j_out = json_pack("{s:[s, s], s:o}",
                "columns", "edge", "value",
                "data", j_data);
    }

finish:

    _arb_vec_clear(x, edge_count);
    _arb_vec_clear(x_out, edge_count);
    _arb_vec_clear(x_initial, edge_count);
    _arb_vec_clear(x_preliminary, edge_count);

    *result_out = result;
    return j_out;
}


json_t *arbplf_second_order_run(void *userdata, json_t *root, int *retcode)
{
    json_t *j_out = NULL;
    model_and_data_t m;
    column_reduction_t r_site;
    int result = 0;
    second_order_query_t query;

    if (!userdata)
    {
        fprintf(stderr, "internal error: unexpected second order query\n");
        result = -1;
        goto finish;
    }

    query = userdata;

    model_and_data_init(m);
    column_reduction_init(r_site);

    result = _parse_second_order(m, r_site, root);
    if (result) goto finish;

    j_out = query(m, r_site, &result);
    if (result) goto finish;

finish:

    *retcode = result;

    column_reduction_clear(r_site);
    model_and_data_clear(m);

    flint_cleanup();
    return j_out;
}
