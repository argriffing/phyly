/*
 * Full precision conditional marginal state distributions at nodes.
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
 * "node_reduction" : {
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
 *  "columns" : ["site", "node", "state", "value"],
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
#include "evaluate_site_marginal.h"
#include "ndaccum.h"
#include "equilibrium.h"
#include "arb_mat_extras.h"
#include "cross_site_ws.h"

#include "parsemodel.h"
#include "parsereduction.h"
#include "runjson.h"
#include "arbplfmarginal.h"


/* Likelihood workspace. */
typedef struct
{
    arb_mat_struct *base_node_vectors;
    arb_mat_struct *lhood_node_vectors;
    arb_mat_struct *lhood_edge_vectors;
    arb_mat_struct *marginal_node_vectors; /* one site, one category */
    arb_mat_struct *cc_marginal_node_vectors; /* one site, across categories */
} likelihood_ws_struct;
typedef likelihood_ws_struct likelihood_ws_t[1];

static void
likelihood_ws_init(likelihood_ws_t w, const model_and_data_t m)
{
    slong node_count = model_and_data_node_count(m);
    slong edge_count = model_and_data_edge_count(m);
    slong state_count = model_and_data_state_count(m);

    w->lhood_edge_vectors = _arb_mat_vec_init(state_count, 1, edge_count);
    w->base_node_vectors = _arb_mat_vec_init(state_count, 1, node_count);
    w->lhood_node_vectors = _arb_mat_vec_init(state_count, 1, node_count);
    w->marginal_node_vectors = _arb_mat_vec_init(state_count, 1, node_count);
    w->cc_marginal_node_vectors = _arb_mat_vec_init(state_count, 1, node_count);
}

static void
likelihood_ws_clear(likelihood_ws_t w, const model_and_data_t m)
{
    slong node_count = model_and_data_node_count(m);
    slong edge_count = model_and_data_edge_count(m);

    _arb_mat_vec_clear(w->lhood_edge_vectors, edge_count);
    _arb_mat_vec_clear(w->base_node_vectors, node_count);
    _arb_mat_vec_clear(w->lhood_node_vectors, node_count);
    _arb_mat_vec_clear(w->marginal_node_vectors, node_count);
    _arb_mat_vec_clear(w->cc_marginal_node_vectors, node_count);
}


static void
_nd_accum_update(nd_accum_t arr,
        likelihood_ws_t w, cross_site_ws_t csw, model_and_data_t m, slong prec)
{
    int site, i, j;
    arb_t cat_lhood, prior_prob, post_lhood, post_lhood_sum;
    nd_axis_struct *site_axis, *node_axis, *state_axis;
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
    node_axis = arr->axes + 1;
    state_axis = arr->axes + 2;

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
            tmat_base = cross_site_ws_transition_matrix(csw, cat, 0);

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
        column_reduction_t r_node,
        column_reduction_t r_state, int *result_out)
{
    json_t * j_out = NULL;
    slong prec;
    int axis_idx;
    nd_axis_struct axes[3];
    nd_accum_t arr;
    cross_site_ws_t csw;
    likelihood_ws_t w;
    int ndim = 3;
    int result = 0;

    slong site_count = model_and_data_site_count(m);
    slong node_count = model_and_data_node_count(m);
    slong state_count = model_and_data_state_count(m);

    /* initialize likelihood workspace */
    cross_site_ws_init(csw, m);
    likelihood_ws_init(w, m);

    /* initialize axes at zero precision */
    nd_axis_init(axes+0, "site", site_count, r_site, 0, NULL, 0);
    nd_axis_init(axes+1, "node", node_count, r_node, 0, NULL, 0);
    nd_axis_init(axes+2, "state", state_count, r_state, 0, NULL, 0);

    /* initialize nd accumulation array */
    nd_accum_pre_init(arr);
    nd_accum_init(arr, axes, ndim);

    /* repeat with increasing precision until there is no precision failure */
    int success = 0;
    for (prec=4; !success; prec <<= 1)
    {
        cross_site_ws_update(csw, m, prec);

        /* recompute axis reduction weights with increased precision */
        nd_axis_update_precision(axes+0, r_site, prec);
        nd_axis_update_precision(axes+1, r_node, prec);
        nd_axis_update_precision(axes+2, r_state, prec);

        /*
         * Recompute the output array with increased working precision.
         * This also updates the workspace conditional and marginal
         * per-node and per-edge likelihood column state vectors.
         */
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
        column_reduction_t r_node,
        column_reduction_t r_state, json_t *root)
{
    json_t *model_and_data = NULL;
    json_t *site_reduction = NULL;
    json_t *node_reduction = NULL;
    json_t *state_reduction = NULL;
    slong site_count, node_count, state_count;
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
                "node_reduction", &node_reduction,
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
    site_count = model_and_data_site_count(m);
    node_count = model_and_data_node_count(m);
    state_count = model_and_data_state_count(m);

    /* validate the site reduction section of the json input */
    result = validate_column_reduction(
            r_site, site_count, "site", site_reduction);
    if (result) return result;

    /* validate the node reduction section of the json input */
    result = validate_column_reduction(
            r_node, node_count, "node", node_reduction);
    if (result) return result;

    /* validate the state reduction section of the json input */
    result = validate_column_reduction(
            r_state, state_count, "state", state_reduction);
    if (result) return result;

    return result;
}


json_t *arbplf_marginal_run(void *userdata, json_t *root, int *retcode)
{
    json_t *j_out = NULL;
    model_and_data_t m;
    column_reduction_t r_site;
    column_reduction_t r_node;
    column_reduction_t r_state;
    int result = 0;

    model_and_data_init(m);
    column_reduction_init(r_site);
    column_reduction_init(r_node);
    column_reduction_init(r_state);

    if (userdata)
    {
        fprintf(stderr, "internal error: unexpected userdata\n");
        result = -1;
        goto finish;
    }

    result = _parse(m, r_site, r_node, r_state, root);
    if (result) goto finish;

    j_out = _query(m, r_site, r_node, r_state, &result);
    if (result) goto finish;

finish:

    *retcode = result;

    model_and_data_clear(m);
    column_reduction_clear(r_site);
    column_reduction_clear(r_node);
    column_reduction_clear(r_state);

    flint_cleanup();
    return j_out;
}
