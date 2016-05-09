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
#include "ndaccum.h"

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
        int *idx_to_a, int *b_to_idx, int site, int rate_category, slong prec)
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

    slong state_count = model_and_data_state_count(m);
    slong edge_count = model_and_data_edge_count(m);

    tmat_base = tmat_collection_entry(csw->transition_matrices,
            rate_category, 0);

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
        /* skip edges that are not requested */
        if (!edge_is_requested[deriv_idx]) continue;

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


static void
_nd_accum_update(nd_accum_t arr,
        likelihood_ws_t w, cross_site_ws_t csw, model_and_data_t m, slong prec)
{
    slong site, idx, edge;
    arb_t cat_rate, cat_lhood, prior_prob, post_lhood, post_lhood_sum;
    nd_axis_struct *site_axis, *edge_axis;
    int *coords;
    slong cat;
    arb_struct *derivatives; /* single category derivatives */
    arb_struct *cc_derivatives; /* cross category derivatives */
    int *idx_to_a = NULL;
    int *b_to_idx = NULL;
    int *idx_to_user_edge = NULL;
    int *edge_idx_is_requested = NULL;

    slong ncats = model_and_data_rate_category_count(m);
    slong site_count = model_and_data_site_count(m);
    slong node_count = model_and_data_node_count(m);
    slong edge_count = model_and_data_edge_count(m);

    site_axis = arr->axes + 0;
    edge_axis = arr->axes + 1;

    /* maps for navigating towards the root of the tree */
    idx_to_a = malloc(edge_count * sizeof(int));
    b_to_idx = malloc(node_count * sizeof(int));
    _csr_graph_get_backward_maps(idx_to_a, b_to_idx, m->g);

    /* map between user visible edge order vs. csr graph edge index */
    idx_to_user_edge = malloc(edge_count * sizeof(int));
    for (edge = 0; edge < edge_count; edge++)
    {
        idx = m->edge_map->order[edge];
        idx_to_user_edge[idx] = edge;
    }

    /* which edges are requested */
    /* todo: this should depend on the site */
    edge_idx_is_requested = calloc(edge_count, sizeof(int));
    for (edge = 0; edge < edge_count; edge++)
    {
        idx = m->edge_map->order[edge];
        edge_idx_is_requested[idx] = edge_axis->request_update[edge];
    }

    derivatives = _arb_vec_init(edge_count);
    cc_derivatives = _arb_vec_init(edge_count);

    arb_init(cat_rate);
    arb_init(cat_lhood);
    arb_init(prior_prob);
    arb_init(post_lhood);
    arb_init(post_lhood_sum);

    coords = malloc(arr->ndim * sizeof(int));

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
        _arb_vec_zero(cc_derivatives, edge_count);

        /*
         * For each category, compute a likelihood for the current site,
         * and compute likelihood derivatives at all requested edges.
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

            evaluate_site_lhood(cat_lhood,
                    w->lhood_node_vectors,
                    w->lhood_edge_vectors,
                    w->base_node_vectors,
                    tmat_base,
                    m->g, m->preorder, node_count, prec);

            /* Compute the likelihood for the site and category. */
            rate_mixture_get_prob(prior_prob, m->rate_mixture, cat, prec);
            arb_mul(post_lhood, prior_prob, cat_lhood, prec);
            arb_add(post_lhood_sum, post_lhood_sum, post_lhood, prec);

            evaluate_site_derivatives(
                    derivatives,
                    edge_idx_is_requested,
                    m, csw, w,
                    idx_to_a, b_to_idx, site, cat, prec);

            rate_mixture_get_rate(cat_rate, m->rate_mixture, cat);

            /* Multiply derivatives by the category rate. */
            for (idx = 0; idx < edge_count; idx++)
            {
                if (!edge_idx_is_requested[idx]) continue;
                arb_mul(derivatives + idx, derivatives + idx, cat_rate, prec);
            }

            /* Accumulate derivatives for this category. */
            for (idx = 0; idx < edge_count; idx++)
            {
                if (!edge_idx_is_requested[idx]) continue;
                arb_addmul(cc_derivatives + idx,
                        derivatives + idx, prior_prob, prec);
            }
        }

        /*
         * Divide accumulated edge derivatives by the site likelihood.
         * d/dx log(a*f(x) + b*g(x)) = (a*f'(x) + b*g'(x)) / (a*f(x) + b*g(x))
         */
        _arb_vec_scalar_div(cc_derivatives,
                cc_derivatives, edge_count, post_lhood_sum, prec);

        /* Update the nd accumulator. */
        for (idx = 0; idx < edge_count; idx++)
        {
            /* skip edges that are not requested */
            if (!edge_idx_is_requested[idx]) continue;

            /* define the coordinate */
            edge = idx_to_user_edge[idx];
            coords[1] = edge;

            /* accumulate */
            nd_accum_accumulate(arr, coords, cc_derivatives + idx, prec);
        }
    }

    _arb_vec_clear(derivatives, edge_count);
    _arb_vec_clear(cc_derivatives, edge_count);

    arb_clear(cat_rate);
    arb_clear(cat_lhood);
    arb_clear(prior_prob);
    arb_clear(post_lhood);
    arb_clear(post_lhood_sum);

    free(coords);
    free(idx_to_a);
    free(b_to_idx);
    free(idx_to_user_edge);
    free(edge_idx_is_requested);
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
