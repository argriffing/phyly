/*
 * Use arbitrary precision matrix operations to compute a log likelihood.
 * The JSON format is used for both input and output.
 * Arbitrary precision is used only internally;
 * double precision floating point without error bounds
 * is used for input and output.
 *
 * The "probability_array" is a semantically flexible structure that defines
 * both the root prior distribution and the observations at the leaves.
 * Each site probability is the sum,
 * over all combinations of state assignments to nodes,
 * of the product of the state probabilities at nodes
 * times the product of transition probabilities on edges.
 *
 * For the log likelihood, the only available selection/aggregation
 * axis is the site axis. So the return value could consist of
 * an array of log likelihoods, or of summed, averaged, or linear
 * combinations of log likelihoods, optionally restricted to a
 * selection of sites.
 *
 * The output should be formatted in a way that is easily
 * readable as a data frame by the python module named pandas as follows:
 * >> import pandas as pd
 * >> f = open('output.json')
 * >> d = pd.read_json(f, orient='split', precise_float=True)
 *
 * On success, a json string is printed to stdout and the process terminates
 * with an exit status of zero.  On failure, information is written to stderr
 * and the process terminates with a nonzero exit status.
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
 * } (optional)
 * }
 *
 * output format (without aggregation of the "site" column):
 * {
 *  "columns" : ["site", "value"],
 *  "data" : [[a, b], [c, d], ..., [y, z]] (# selected sites)
 * }
 *
 * output format (with aggregation of the "site" column):
 * {
 *  "columns" : ["value"],
 *  "data" : [a]
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
#include "arbplfll.h"


/* This object changes in the course of the iteration over sites. */
typedef struct
{
    arb_mat_struct *base_node_vectors;
    arb_mat_struct *lhood_node_vectors;
} likelihood_ws_struct;
typedef likelihood_ws_struct likelihood_ws_t[1];

static void
likelihood_ws_init(likelihood_ws_t w, const model_and_data_t m)
{
    slong node_count = model_and_data_node_count(m);
    slong state_count = model_and_data_state_count(m);

    w->base_node_vectors = _arb_mat_vec_init(state_count, 1, node_count);
    w->lhood_node_vectors = _arb_mat_vec_init(state_count, 1, node_count);
}

static void
likelihood_ws_clear(likelihood_ws_t w, const model_and_data_t m)
{
    slong node_count = model_and_data_node_count(m);

    _arb_mat_vec_clear(w->base_node_vectors, node_count);
    _arb_mat_vec_clear(w->lhood_node_vectors, node_count);
}

/* this function uses arbplfmarginal as a template */
static void
_nd_accum_update(nd_accum_t arr,
        likelihood_ws_t w, cross_site_ws_t csw, model_and_data_t m, slong prec)
{
    slong site, cat;
    arb_t cat_lhood, ll, post_lhood_sum;
    nd_axis_struct *site_axis;
    int *coords;

    slong ncats = model_and_data_rate_category_count(m);
    slong site_count = model_and_data_site_count(m);

    arb_init(cat_lhood);
    arb_init(ll);
    arb_init(post_lhood_sum);

    coords = malloc(arr->ndim * sizeof(int));

    site_axis = arr->axes + 0;

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
        pmat_update_base_node_vectors(w->base_node_vectors, m->p, site);

        /* sum over rate categories */
        arb_zero(post_lhood_sum);
        for (cat = 0; cat < ncats; cat++)
        {
            const arb_struct * prior_prob = csw->rate_mix_prior + cat;
            const arb_mat_struct * tmat_base;
            tmat_base = cross_site_ws_transition_matrix(csw, cat, 0);

            evaluate_site_lhood(cat_lhood,
                    w->lhood_node_vectors,
                    NULL,
                    w->base_node_vectors,
                    m->root_prior, csw->equilibrium,
                    tmat_base,
                    m->g, m->navigation->preorder, csw->node_count, prec);

            /* Compute the likelihood for the site and category. */
            arb_addmul(post_lhood_sum, prior_prob, cat_lhood, prec);
        }

        arb_log(ll, post_lhood_sum, prec);
        nd_accum_accumulate(arr, coords, ll, prec);
    }

    arb_clear(cat_lhood);
    arb_clear(post_lhood_sum);
    arb_clear(ll);

    free(coords);
}

/* this function uses arbplfmarginal as a template */
static json_t *
_query(model_and_data_t m, column_reduction_t r_site, int *result_out)
{
    json_t * j_out = NULL;
    slong prec;
    int axis_idx;
    nd_axis_struct axes[1];
    nd_accum_t arr;
    cross_site_ws_t csw;
    likelihood_ws_t w;
    int ndim = 1;
    int result = 0;

    slong site_count = model_and_data_site_count(m);

    /* initialize likelihood workspace */
    cross_site_ws_init(csw, m);
    likelihood_ws_init(w, m);

    /* initialize axes at zero precision */
    nd_axis_init(axes+0, "site", site_count, r_site, 0, NULL, 0);

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
_parse(model_and_data_t m, column_reduction_t r_site, json_t *root)
{
    json_t *model_and_data = NULL;
    json_t *site_reduction = NULL;
    slong site_count;
    int result = 0;

    /* unpack the top level of json input */
    {
        size_t flags;
        json_error_t err;
        flags = JSON_STRICT;
        result = json_unpack_ex(root, &err, flags,
                "{s:o, s?o}",
                "model_and_data", &model_and_data,
                "site_reduction", &site_reduction
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

    /* validate the site reduction section of the json input */
    result = validate_column_reduction(
            r_site, site_count, "site", site_reduction);
    if (result) return result;

    return result;
}


json_t *arbplf_ll_run(void *userdata, json_t *root, int *retcode)
{
    json_t *j_out = NULL;
    model_and_data_t m;
    column_reduction_t r_site;
    int result = 0;

    model_and_data_init(m);
    column_reduction_init(r_site);

    if (userdata)
    {
        fprintf(stderr, "internal error: unexpected userdata\n");
        result = -1;
        goto finish;
    }

    result = _parse(m, r_site, root);
    if (result) goto finish;

    j_out = _query(m, r_site, &result);
    if (result) goto finish;

finish:

    *retcode = result;

    model_and_data_clear(m);
    column_reduction_clear(r_site);

    flint_cleanup();
    return j_out;
}
