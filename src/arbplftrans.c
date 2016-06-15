/*
 * Full precision labeled transition count expectations.
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
 *  "aggregation" : {"sum" | "avg" | [a, b, c, ...]} (optional)
 * },
 * "edge_reduction" : {
 *  "selection" : [a, b, c, ...], (optional)
 *  "aggregation" : {"sum" | "avg" | [a, b, c, ...]} (optional)
 * },
 * "trans_reduction" : {
 *  "selection" : [[a, b], [c, d], ...], (optional)
 *  "aggregation" : {"sum" | "avg" | [a, b, c, ...]} (optional)
 * }}
 *
 * output format:
 * {
 *  "columns" : ["site", "edge", "first_state", "second_state", "value"],
 *  "data" : [[. . . . .], [. . . . .], ..., [. . . . .]]
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
#include "cross_site_ws.h"
#include "arb_mat_extras.h"

#include "parsemodel.h"
#include "parsereduction.h"
#include "runjson.h"
#include "arbplftrans.h"

#define SITE_AXIS 0
#define EDGE_AXIS 1
#define TRANS_AXIS 2

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
    w->lhood_node_vectors = _arb_mat_vec_init(state_count, 1, node_count);
    w->lhood_edge_vectors = _arb_mat_vec_init(state_count, 1, edge_count);
    w->forward_node_vectors = _arb_mat_vec_init(state_count, 1, node_count);
    w->forward_edge_vectors = _arb_mat_vec_init(state_count, 1, edge_count);
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
_update_state_pair_frechet_matrices(
        cross_site_ws_t csw, model_and_data_t m,
        nd_axis_struct *edge_axis,
        slong first_state, slong second_state, slong prec)
{
    arb_mat_t P, L, Q;
    slong cat;
    arb_t rate;

    slong state_count = model_and_data_state_count(m);
    slong edge_count = model_and_data_edge_count(m);
    slong rate_category_count = model_and_data_rate_category_count(m);

    arb_init(rate);
    arb_mat_init(P, state_count, state_count);
    arb_mat_init(L, state_count, state_count);
    arb_mat_init(Q, state_count, state_count);
    arb_set(arb_mat_entry(L, first_state, second_state),
            arb_mat_entry(csw->rate_matrix, first_state, second_state));

    for (cat = 0; cat < rate_category_count; cat++)
    {
        slong edge;
        const arb_struct * cat_rate = csw->rate_mix_rates + cat;
        for (edge = 0; edge < edge_count; edge++)
        {
            slong idx = m->edge_map->order[edge];
            const arb_struct * edge_rate = csw->edge_rates + idx;
            arb_mat_struct *fmat;
            if (!edge_axis->request_update[edge]) continue;
            fmat = cross_site_ws_trans_frechet_matrix(csw, cat, idx);
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
        nd_axis_struct *edge_axis, nd_axis_struct *trans_axis,
        const int *first_idx, const int *second_idx, slong prec)
{
    arb_mat_t P, L, Q;
    arb_t rate;
    slong first_state, second_state;
    slong cat, trans_idx;

    slong state_count = model_and_data_state_count(m);
    slong edge_count = model_and_data_edge_count(m);
    slong rate_category_count = model_and_data_rate_category_count(m);

    arb_init(rate);
    arb_mat_init(P, state_count, state_count);
    arb_mat_init(L, state_count, state_count);
    arb_mat_init(Q, state_count, state_count);

    /* set entries of L to the requested transition weights */
    for (trans_idx = 0; trans_idx < trans_axis->n; trans_idx++)
    {
        first_state = first_idx[trans_idx];
        second_state = second_idx[trans_idx];
        arb_add(arb_mat_entry(L, first_state, second_state),
                arb_mat_entry(L, first_state, second_state),
                trans_axis->agg_weights + trans_idx, prec);
    }

    /* multiply entries of L by the rate matrix entries */
    arb_mat_mul_entrywise(L, L, csw->rate_matrix, prec);

    /* divide L by the global weight divisor */
    arb_mat_scalar_div_arb(L, L, trans_axis->agg_weight_divisor, prec);

    for (cat = 0; cat < rate_category_count; cat++)
    {
        slong edge;
        const arb_struct * cat_rate = csw->rate_mix_rates + cat;
        for (edge = 0; edge < edge_count; edge++)
        {
            slong idx = m->edge_map->order[edge];
            const arb_struct * edge_rate = csw->edge_rates + idx;
            arb_mat_struct *fmat;
            if (!edge_axis->request_update[edge]) continue;
            fmat = cross_site_ws_trans_frechet_matrix(csw, cat, idx);
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

/*
 * Note that the expectations are multiplied by the rates.
 * This is a difference from the analogous function in arbplfdwell.c.
 */
static void
_update_site(nd_accum_t arr,
        likelihood_ws_t w, cross_site_ws_t csw, model_and_data_t m,
        int *coords, slong site, slong prec)
{
    int edge, idx, cat;
    arb_t site_lhood, cat_lhood, lhood;
    arb_t tmp;

    slong state_count = model_and_data_state_count(m);
    slong edge_count = model_and_data_edge_count(m);
    slong node_count = model_and_data_node_count(m);
    slong rate_category_count = model_and_data_rate_category_count(m);

    arb_init(site_lhood);
    arb_init(cat_lhood);
    arb_init(lhood);
    arb_init(tmp);

    /* only the edge axis is handled at this depth */
    nd_axis_struct *edge_axis = arr->axes + EDGE_AXIS;

    /* update base node vectors */
    pmat_update_base_node_vectors(w->base_node_vectors, m->p, site);

    /* clear cross-category expectations and site lhood */
    _arb_vec_zero(w->cc_edge_expectations, edge_count);
    arb_zero(site_lhood);

    for (cat = 0; cat < rate_category_count; cat++)
    {
        const arb_struct * cat_rate = csw->rate_mix_rates + cat;
        const arb_struct * prior_prob = csw->rate_mix_prior + cat;
        arb_mat_struct *tmat_base, *fmat_base;
        tmat_base = cross_site_ws_transition_matrix(csw, cat, 0);
        fmat_base = cross_site_ws_trans_frechet_matrix(csw, cat, 0);

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

            /*
             * Multiply by the product of the category rate,
             * the edge rate, and the prior category probability.
             * In the analogous 'dwell' function (as opposed to 'trans'),
             * only the prior category probability is included.
             */
            arb_mul(tmp, cat_rate, csw->edge_rates + idx, prec);
            arb_mul(tmp, tmp, prior_prob, prec);
            arb_addmul(w->cc_edge_expectations + idx,
                       w->edge_expectations + idx, tmp, prec);
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
    arb_clear(tmp);
}

static void
_nd_accum_update_state_agg(nd_accum_t arr,
        likelihood_ws_t w, cross_site_ws_t csw, model_and_data_t m,
        const int *first_idx, const int *second_idx, slong prec)
{
    slong site;
    int *coords;

    slong site_count = model_and_data_site_count(m);

    nd_axis_struct *site_axis = arr->axes + SITE_AXIS;
    nd_axis_struct *edge_axis = arr->axes + EDGE_AXIS;
    nd_axis_struct *trans_axis = arr->axes + TRANS_AXIS;

    coords = malloc(arr->ndim * sizeof(int));

    /* zero all requested cells of the array */
    nd_accum_zero_requested_cells(arr);

    /*
     * Update the frechet matrix for each edge.
     * At this point the rate matrix has been normalized
     * to have zero row sums, but it has not been scaled
     * by the edge rate coefficients.
     */
    _update_aggregated_state_frechet_matrices(
            csw, m, edge_axis, trans_axis, first_idx, second_idx, prec);

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
        likelihood_ws_t w, cross_site_ws_t csw, model_and_data_t m,
        const int *first_idx, const int *second_idx, slong prec)
{
    slong site, trans_idx, first_state, second_state;
    int *coords;

    slong site_count = model_and_data_site_count(m);

    nd_axis_struct *site_axis = arr->axes + SITE_AXIS;
    nd_axis_struct *edge_axis = arr->axes + EDGE_AXIS;
    nd_axis_struct *trans_axis = arr->axes + TRANS_AXIS;

    coords = malloc(arr->ndim * sizeof(int));

    /* zero all requested cells of the array */
    nd_accum_zero_requested_cells(arr);

    /*
     * Update the output array at the given precision.
     * Axes have already been updated for this precision.
     * The nd array links to axis selection and aggregation information,
     * and the model and data are provided separately.
     */
    for (trans_idx = 0; trans_idx < trans_axis->n; trans_idx++)
    {
        if (!trans_axis->request_update[trans_idx]) continue;
        coords[TRANS_AXIS] = trans_idx;

        first_state = first_idx[trans_idx];
        second_state = second_idx[trans_idx];

        _update_state_pair_frechet_matrices(
                csw, m, edge_axis, first_state, second_state, prec);

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
        column_reduction_t r_trans,
        int *first_idx,
        int *second_idx,
        int *result_out)
{
    json_t * j_out = NULL;
    slong prec;
    int ndim, axis_idx;
    nd_axis_struct axes[3];
    struct nd_component_axis axis_components[2];
    nd_accum_t arr;
    likelihood_ws_t w;
    cross_site_ws_t csw;
    int result = 0;
    int success;

    slong site_count = model_and_data_site_count(m);
    slong edge_count = model_and_data_edge_count(m);

    nd_axis_struct *site_axis = axes + SITE_AXIS;
    nd_axis_struct *edge_axis = axes + EDGE_AXIS;
    nd_axis_struct *trans_axis = axes + TRANS_AXIS;

    /* initialize likelihood workspace */
    cross_site_ws_init(csw, m);
    cross_site_ws_init_trans(csw);
    likelihood_ws_init(w, m);

    axis_components[0].name = "first_state";
    axis_components[0].indices = first_idx;
    axis_components[1].name = "second_state";
    axis_components[1].indices = second_idx;

    /* initialize the first two axes at zero precision */
    nd_axis_init(site_axis, "site", site_count, r_site, 0, NULL, 0);
    nd_axis_init(edge_axis, "edge", edge_count, r_edge, 0, NULL, 0);

    nd_axis_init(trans_axis, "trans", r_trans->selection_len, r_trans,
            2, axis_components, 0);

    /*
     * Define the number of axes to use in the nd accumulator.
     * The reason for this complication is that state aggregation
     * can be done more efficiently using linear algebra tricks instead
     * of using the generic framework.
     */
    ndim = (r_trans->agg_mode == AGG_NONE) ? 3 : 2;

    /* initialize nd accumulation array */
    nd_accum_pre_init(arr);
    nd_accum_init(arr, axes, ndim);

    /* repeat with increasing precision until there is no precision failure */
    for (success=0, prec=4; !success; prec <<= 1)
    {
        cross_site_ws_update(csw, m, prec);

        /* Recompute axis reduction weights with increased precision. */
        nd_axis_update_precision(site_axis, r_site, prec);
        nd_axis_update_precision(edge_axis, r_edge, prec);
        nd_axis_update_precision(trans_axis, r_trans, prec);

        /*
         * Recompute the output array with increased working precision.
         * This also updates the workspace conditional and marginal
         * per-node and per-edge likelihood column state vectors.
         */
        if (r_trans->agg_mode == AGG_NONE)
        {
            _nd_accum_update(arr, w, csw, m, first_idx, second_idx, prec);
        }
        else
        {
            _nd_accum_update_state_agg(arr, w, csw, m, first_idx, second_idx, prec);
        }

        /* check whether entries are accurate to full relative precision  */
        success = nd_accum_can_round(arr);
    }

    /* build the json output using the nd array */
    j_out = nd_accum_get_json(arr, &result);
    if (result) goto finish;

finish:

    likelihood_ws_clear(w, m);
    cross_site_ws_clear(csw);

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
        column_reduction_t r_trans,
        int **first_idx, int **second_idx,
        json_t *root)
{
    json_t *model_and_data = NULL;
    json_t *site_reduction = NULL;
    json_t *edge_reduction = NULL;
    json_t *trans_reduction = NULL;
    slong site_count, edge_count, state_count;
    int result = 0;

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
                "trans_reduction", &trans_reduction
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
    state_count = model_and_data_state_count(m);

    /* validate the site reduction section of the json input */
    result = validate_column_reduction(
            r_site, site_count, "site", site_reduction);
    if (result) return result;

    /* validate the edge reduction section of the json input */
    result = validate_column_reduction(
            r_edge, edge_count, "edge", edge_reduction);
    if (result) return result;

    /* validate the state reduction section of the json input */
    result = validate_column_pair_reduction(r_trans, first_idx, second_idx,
            state_count, state_count, "trans", trans_reduction);
    if (result) return result;

    return result;
}


json_t *arbplf_trans_run(void *userdata, json_t *root, int *retcode)
{
    json_t *j_out = NULL;
    model_and_data_t m;
    column_reduction_t r_site;
    column_reduction_t r_edge;
    column_reduction_t r_trans;
    int *first_idx;
    int *second_idx;
    int result = 0;

    first_idx = NULL;
    second_idx = NULL;

    model_and_data_init(m);
    column_reduction_init(r_site);
    column_reduction_init(r_edge);
    column_reduction_init(r_trans);

    if (userdata)
    {
        fprintf(stderr, "internal error: unexpected userdata\n");
        result = -1;
        goto finish;
    }

    result = _parse(m, r_site, r_edge, r_trans, &first_idx, &second_idx, root);
    if (result) goto finish;

    j_out = _query(m, r_site, r_edge, r_trans, first_idx, second_idx, &result);
    if (result) goto finish;

finish:

    *retcode = result;

    model_and_data_clear(m);
    column_reduction_clear(r_site);
    column_reduction_clear(r_edge);
    column_reduction_clear(r_trans);

    flint_free(first_idx);
    flint_free(second_idx);

    flint_cleanup();
    return j_out;
}
