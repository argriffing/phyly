/*
 * Full precision conditional marginal state distributions on edges.
 *
 * The JSON format is used for both input and output.
 * Arbitrary precision is used only internally;
 * double precision floating point without error bounds
 * is used for input and output.
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
 * },
 * "edge_reduction" : {
 *  "selection" : [a, b, c, ...], (optional)
 *  "aggregation" : {"sum" | "avg" | [a, b, c, ...]}
 * },
 * "state_reduction" : {
 *  "selection" : [a, b, c, ...], (optional)
 *  "aggregation" : {"sum" | "avg" | [a, b, c, ...]}
 * }}
 *
 * output format without aggregation:
 * {
 *  "columns" : ["site", "edge", "state", "value"],
 *  "data" : [[a, b, c, d], [e, f, g, h], ..., [w, x, y, z]]
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
#include "evaluate_site_forward.h"
#include "evaluate_site_frechet.h"
#include "ndaccum.h"
#include "equilibrium.h"
#include "arb_mat_extras.h"
#include "cross_site_ws.h"

#include "parsemodel.h"
#include "parsereduction.h"
#include "runjson.h"
#include "arbplfdwell.h"

#define SITE_AXIS 0
#define EDGE_AXIS 1
#define STATE_AXIS 2


typedef struct
{
    arb_struct *edge_expectations;
    arb_struct *cc_edge_expectations;
    arb_mat_struct *base_node_vectors;
    arb_mat_struct *lhood_node_vectors;
    arb_mat_struct *lhood_edge_vectors;
    arb_mat_struct *forward_node_vectors;
    arb_mat_struct *forward_edge_vectors;
} likelihood_ws_struct;
typedef likelihood_ws_struct likelihood_ws_t[1];

static void
likelihood_ws_init(likelihood_ws_t w, const model_and_data_t m)
{
    slong node_count = model_and_data_node_count(m);
    slong edge_count = model_and_data_edge_count(m);
    slong state_count = model_and_data_state_count(m);

    w->edge_expectations = _arb_vec_init(edge_count);
    w->cc_edge_expectations = _arb_vec_init(edge_count);
    w->base_node_vectors = _arb_mat_vec_init(state_count, 1, node_count);
    w->lhood_edge_vectors = _arb_mat_vec_init(state_count, 1, edge_count);
    w->lhood_node_vectors = _arb_mat_vec_init(state_count, 1, node_count);
    w->forward_edge_vectors = _arb_mat_vec_init(state_count, 1, edge_count);
    w->forward_node_vectors = _arb_mat_vec_init(state_count, 1, node_count);
}

static void
likelihood_ws_clear(likelihood_ws_t w, const model_and_data_t m)
{
    slong node_count = model_and_data_node_count(m);
    slong edge_count = model_and_data_edge_count(m);

    _arb_vec_clear(w->edge_expectations, edge_count);
    _arb_vec_clear(w->cc_edge_expectations, edge_count);
    _arb_mat_vec_clear(w->base_node_vectors, node_count);
    _arb_mat_vec_clear(w->lhood_edge_vectors, edge_count);
    _arb_mat_vec_clear(w->lhood_node_vectors, node_count);
    _arb_mat_vec_clear(w->forward_edge_vectors, edge_count);
    _arb_mat_vec_clear(w->forward_node_vectors, node_count);
}

/*
 * Update the frechet matrix for each rate category and edge.
 * At this point the rate matrix has been normalized
 * to have zero row sums, but it has not been scaled
 * by the edge rate coefficients.
 * The frechet matrices must already have been initialized.
 */
static void
_update_single_state_frechet_matrices(
        cross_site_ws_t csw, model_and_data_t m,
        slong state, slong prec)
{
    slong cat, idx;
    arb_mat_t P, L, Q;
    arb_t rate;

    slong state_count = model_and_data_state_count(m);
    slong edge_count = model_and_data_edge_count(m);
    slong rate_category_count = model_and_data_rate_category_count(m);

    arb_init(rate);
    arb_mat_init(P, state_count, state_count);
    arb_mat_init(L, state_count, state_count);
    arb_mat_init(Q, state_count, state_count);
    arb_one(arb_mat_entry(L, state, state));
    for (cat = 0; cat < rate_category_count; cat++)
    {
        const arb_struct *cat_rate = csw->rate_mix_rates + cat;
        for (idx = 0; idx < edge_count; idx++)
        {
            const arb_struct *edge_rate = csw->edge_rates + idx;
            arb_mat_struct *fmat;
            fmat = cross_site_ws_dwell_frechet_matrix(csw, cat, idx);
            arb_mul(rate, edge_rate, cat_rate, prec);
            arb_mat_scalar_mul_arb(Q, csw->rate_matrix, rate, prec);
            _arb_mat_exp_frechet(P, fmat, Q, L, prec);
        }
    }
    arb_clear(rate);
    arb_mat_clear(P);
    arb_mat_clear(L);
    arb_mat_clear(Q);
}

static void
_update_aggregated_state_frechet_matrices(
        cross_site_ws_t csw, model_and_data_t m,
        nd_axis_struct *state_axis, slong prec)
{
    slong state, cat, idx;
    arb_mat_t P, L, Q;
    arb_t rate;

    slong state_count = model_and_data_state_count(m);
    slong edge_count = model_and_data_edge_count(m);
    slong rate_category_count = model_and_data_rate_category_count(m);

    arb_init(rate);
    arb_mat_init(P, state_count, state_count);
    arb_mat_init(L, state_count, state_count);
    arb_mat_init(Q, state_count, state_count);
    for (state = 0; state < state_count; state++)
    {
        arb_div(arb_mat_entry(L, state, state),
                state_axis->agg_weights + state,
                state_axis->agg_weight_divisor, prec);
    }
    for (cat = 0; cat < rate_category_count; cat++)
    {
        const arb_struct *cat_rate = csw->rate_mix_rates + cat;
        for (idx = 0; idx < edge_count; idx++)
        {
            const arb_struct *edge_rate = csw->edge_rates + idx;
            arb_mat_struct *fmat;
            fmat = cross_site_ws_dwell_frechet_matrix(csw, cat, idx);
            arb_mul(rate, edge_rate, cat_rate, prec);
            arb_mat_scalar_mul_arb(Q, csw->rate_matrix, rate, prec);
            _arb_mat_exp_frechet(P, fmat, Q, L, prec);
        }
    }
    arb_clear(rate);
    arb_mat_clear(P);
    arb_mat_clear(L);
    arb_mat_clear(Q);
}

static void
_update_site(nd_accum_t arr,
        likelihood_ws_t w, cross_site_ws_t csw, model_and_data_t m,
        int *coords, slong site, slong prec)
{
    int edge, idx, cat;
    arb_t site_lhood, cat_lhood, lhood;

    slong state_count = model_and_data_state_count(m);
    slong edge_count = model_and_data_edge_count(m);
    slong node_count = model_and_data_node_count(m);
    slong rate_category_count = model_and_data_rate_category_count(m);

    arb_init(site_lhood);
    arb_init(cat_lhood);
    arb_init(lhood);

    /* only the edge axis is handled at this depth */
    nd_axis_struct *edge_axis = arr->axes + EDGE_AXIS;

    /* update base node vectors */
    pmat_update_base_node_vectors(w->base_node_vectors, m->p, site);

    /* clear cross-category expectations and site lhood */
    _arb_vec_zero(w->cc_edge_expectations, edge_count);
    arb_zero(site_lhood);

    for (cat = 0; cat < rate_category_count; cat++)
    {
        const arb_struct *prior_prob = csw->rate_mix_prior + cat;
        arb_mat_struct *tmat_base, *fmat_base;
        tmat_base = cross_site_ws_transition_matrix(csw, cat, 0);
        fmat_base = cross_site_ws_dwell_frechet_matrix(csw, cat, 0);

        /*
         * Update per-node and per-edge likelihood vectors.
         * Actually the likelihood vectors on edges are not used.
         * This is a backward pass from the leaves to the root.
         */
        evaluate_site_lhood(lhood,
                w->lhood_node_vectors,
                w->lhood_edge_vectors,
                w->base_node_vectors,
                m->root_prior, csw->equilibrium,
                tmat_base,
                m->g, m->navigation->preorder, node_count, prec);

        /* Update forward vectors. */
        evaluate_site_forward(
                w->forward_edge_vectors,
                w->forward_node_vectors,
                w->base_node_vectors,
                w->lhood_edge_vectors,
                m->root_prior, csw->equilibrium,
                tmat_base, m->g, m->navigation,
                csw->node_count, csw->state_count, prec);

        /* Update expectations at edges. */
        evaluate_site_frechet(
                w->edge_expectations,
                w->lhood_node_vectors,
                w->forward_edge_vectors,
                fmat_base,
                m->g, m->navigation->preorder, node_count, state_count, prec);

        /* compute category likelihood */
        arb_mul(cat_lhood, lhood, prior_prob, prec);
        arb_add(site_lhood, site_lhood, cat_lhood, prec);

        /* Accumulate cross-category expectations. */
        for (edge = 0; edge < edge_count; edge++)
        {
            if (!edge_axis->request_update[edge]) continue;
            idx = m->edge_map->order[edge];
            arb_addmul(w->cc_edge_expectations + idx,
                       w->edge_expectations + idx, prior_prob, prec);
        }
    }

    /* Divide cross-category expectations by site lhood */
    for (edge = 0; edge < edge_count; edge++)
    {
        if (!edge_axis->request_update[edge]) continue;
        idx = m->edge_map->order[edge];
        arb_div(w->cc_edge_expectations + idx,
                w->cc_edge_expectations + idx,
                site_lhood, prec);
    }

    /* Update the nd accumulator. */
    for (edge = 0; edge < edge_count; edge++)
    {
        /* skip edges that are not requested */
        if (!edge_axis->request_update[edge]) continue;
        coords[EDGE_AXIS] = edge;

        /*
         * Accumulate.
         * Note that the axes, accumulator, and json interface
         * work with "user" edge indices,
         * whereas the workspace arrays work with
         * tree graph preorder edge indices.
         */
        idx = m->edge_map->order[edge];
        nd_accum_accumulate(arr,
                coords, w->cc_edge_expectations + idx, prec);
    }

    arb_clear(site_lhood);
    arb_clear(cat_lhood);
    arb_clear(lhood);
}


static void
_nd_accum_update_state_agg(nd_accum_t arr,
        likelihood_ws_t w, cross_site_ws_t csw, model_and_data_t m, slong prec)
{
    slong site;
    int *coords;

    slong site_count = model_and_data_site_count(m);

    nd_axis_struct *site_axis = arr->axes + SITE_AXIS;
    nd_axis_struct *state_axis = arr->axes + STATE_AXIS;

    coords = malloc(arr->ndim * sizeof(int));

    /* zero all requested cells of the array */
    nd_accum_zero_requested_cells(arr);

    /* todo: what about zero branch length */

    /*
     * Update the frechet matrix for each edge.
     * At this point the rate matrix has been normalized
     * to have zero row sums, but it has not been scaled
     * by the edge rate coefficients.
     */
    _update_aggregated_state_frechet_matrices(csw, m, state_axis, prec);

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
        coords[SITE_AXIS] = site;

        _update_site(arr, w, csw, m, coords, site, prec);
    }

    free(coords);
}


static void
_nd_accum_update(nd_accum_t arr,
        likelihood_ws_t w, cross_site_ws_t csw, model_and_data_t m, slong prec)
{
    int state, site;
    nd_axis_struct *state_axis, *site_axis;
    int *coords;

    slong state_count = model_and_data_state_count(m);
    slong site_count = model_and_data_site_count(m);

    coords = malloc(arr->ndim * sizeof(int));

    site_axis = arr->axes + SITE_AXIS;
    state_axis = arr->axes + STATE_AXIS;

    /* zero all requested cells of the array */
    nd_accum_zero_requested_cells(arr);

    /*
     * Update the output array at the given precision.
     * Axes have already been updated for this precision.
     * The nd array links to axis selection and aggregation information,
     * and the model and data are provided separately.
     */
    for (state = 0; state < state_count; state++)
    {
        if (!state_axis->request_update[state]) continue;
        coords[STATE_AXIS] = state;

        /*
         * Update the frechet matrix for each rate category and edge.
         * At this point the rate matrix has been normalized
         * to have zero row sums, but it has not been scaled
         * by the edge rate coefficients.
         */
        _update_single_state_frechet_matrices(csw, m, state, prec);

        for (site = 0; site < site_count; site++)
        {
            /* skip sites that are not requested */
            if (!site_axis->request_update[site]) continue;
            coords[SITE_AXIS] = site;

            _update_site(arr, w, csw, m, coords, site, prec);
        }
    }

    free(coords);
}


static json_t *
_query(model_and_data_t m,
        column_reduction_t r_site,
        column_reduction_t r_edge,
        column_reduction_t r_state,
        int *result_out)
{
    json_t * j_out = NULL;
    slong prec;
    int ndim, axis_idx;
    nd_axis_struct axes[3];
    nd_accum_t arr;
    likelihood_ws_t w;
    cross_site_ws_t csw;
    int result = 0;

    slong state_count = model_and_data_state_count(m);
    slong site_count = model_and_data_site_count(m);
    slong edge_count = model_and_data_edge_count(m);

    nd_axis_struct *site_axis = axes + SITE_AXIS;
    nd_axis_struct *edge_axis = axes + EDGE_AXIS;
    nd_axis_struct *state_axis = axes + STATE_AXIS;

    /* initialize likelihood workspace */
    cross_site_ws_init(csw, m);
    cross_site_ws_init_dwell(csw);
    likelihood_ws_init(w, m);

    /* initialize axes at zero precision */
    nd_axis_init(site_axis, "site", site_count, r_site, 0, NULL, 0);
    nd_axis_init(edge_axis, "edge", edge_count, r_edge, 0, NULL, 0);
    nd_axis_init(state_axis, "state", state_count, r_state, 0, NULL, 0);

    /*
     * Define the number of axes to use in the nd accumulator.
     * The reason for this complication is that state aggregation
     * can be done more efficiently using linear algebra tricks instead
     * of using the generic framework.
     */
    ndim = (r_state->agg_mode == AGG_NONE) ? 3 : 2;

    /* initialize nd accumulation array */
    nd_accum_pre_init(arr);
    nd_accum_init(arr, axes, ndim);

    /* repeat with increasing precision until there is no precision failure */
    int success = 0;
    for (prec=4; !success; prec <<= 1)
    {
        cross_site_ws_update(csw, m, prec);

        /* Recompute axis reduction weights with increased precision. */
        nd_axis_update_precision(site_axis, r_site, prec);
        nd_axis_update_precision(edge_axis, r_edge, prec);
        nd_axis_update_precision(state_axis, r_state, prec);

        /* Recompute the output array with increased working precision. */
        if (r_state->agg_mode == AGG_NONE)
        {
            _nd_accum_update(arr, w, csw, m, prec);
        }
        else
        {
            _nd_accum_update_state_agg(arr, w, csw, m, prec);
        }

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
    for (axis_idx = 0; axis_idx < 3; axis_idx++)
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
        column_reduction_t r_edge,
        column_reduction_t r_state, json_t *root)
{
    json_t *model_and_data = NULL;
    json_t *site_reduction = NULL;
    json_t *edge_reduction = NULL;
    json_t *state_reduction = NULL;
    slong state_count, site_count, edge_count;
    int result;

    result = 0;

    /* unpack the top level of json input */
    {
        size_t flags;
        json_error_t err;
        flags = JSON_STRICT;
        result = json_unpack_ex(root, &err, flags,
                "{s:o, s?o, s?o, s?o}",
                "model_and_data", &model_and_data,
                "site_reduction", &site_reduction,
                "edge_reduction", &edge_reduction,
                "state_reduction", &state_reduction
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
    state_count = model_and_data_state_count(m);
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

    /* validate the state reduction section of the json input */
    result = validate_column_reduction(
            r_state, state_count, "state", state_reduction);
    if (result) return result;

    return result;
}


json_t *arbplf_dwell_run(void *userdata, json_t *root, int *retcode)
{
    json_t *j_out = NULL;
    model_and_data_t m;
    column_reduction_t r_site;
    column_reduction_t r_edge;
    column_reduction_t r_state;
    int result = 0;

    model_and_data_init(m);
    column_reduction_init(r_site);
    column_reduction_init(r_edge);
    column_reduction_init(r_state);

    if (userdata)
    {
        fprintf(stderr, "internal error: unexpected userdata\n");
        result = -1;
        goto finish;
    }

    result = _parse(m, r_site, r_edge, r_state, root);
    if (result) goto finish;

    j_out = _query(m, r_site, r_edge, r_state, &result);
    if (result) goto finish;

finish:

    *retcode = result;

    model_and_data_clear(m);
    column_reduction_clear(r_site);
    column_reduction_clear(r_edge);
    column_reduction_clear(r_state);

    flint_cleanup();
    return j_out;
}
