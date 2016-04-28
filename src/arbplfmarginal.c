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

#include "parsemodel.h"
#include "parsereduction.h"
#include "runjson.h"
#include "arbplfmarginal.h"


/* Likelihood workspace. */
typedef struct
{
    int node_count;
    int edge_count;
    int state_count;
    arb_struct *edge_rates;
    arb_struct *equilibrium;
    arb_mat_t rate_matrix;
    arb_mat_struct *transition_matrices;
    arb_mat_struct *base_node_vectors;
    arb_mat_struct *lhood_node_vectors;
    arb_mat_struct *lhood_edge_vectors;
    arb_mat_struct *marginal_node_vectors;
} likelihood_ws_struct;
typedef likelihood_ws_struct likelihood_ws_t[1];

static void
likelihood_ws_init(likelihood_ws_t w, model_and_data_t m)
{
    csr_graph_struct *g;
    int i;
    arb_mat_struct *tmat;
    double tmpd;

    /*
     * This is the csr graph index of edge (a, b).
     * Given this index, node b is directly available
     * from the csr data structure.
     * The rate coefficient associated with the edge will also be available.
     * On the other hand, the index of node 'a' will be available through
     * the iteration order rather than directly from the index.
     */
    int idx;

    g = m->g;

    w->node_count = g->n;
    w->edge_count = g->nnz;
    w->state_count = arb_mat_nrows(m->mat);

    w->edge_rates = _arb_vec_init(w->edge_count);
    arb_mat_init(w->rate_matrix, w->state_count, w->state_count);
    w->transition_matrices = flint_malloc(
            w->edge_count * sizeof(arb_mat_struct));
    w->equilibrium = NULL;
    if (model_and_data_uses_equilibrium(m))
    {
        w->equilibrium = _arb_vec_init(w->state_count);
    }

    /* intialize transition probability matrices */
    for (idx = 0; idx < w->edge_count; idx++)
    {
        tmat = w->transition_matrices + idx;
        arb_mat_init(tmat, w->state_count, w->state_count);
    }

    /*
     * Define the map from csr edge index to edge rate.
     * The edge rate is represented in arbitrary precision,
     * but is assumed to take exactly the double precision input value.
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

    /* initialize per-node state vectors */
    w->base_node_vectors = flint_malloc(
            w->node_count * sizeof(arb_mat_struct));
    w->lhood_node_vectors = flint_malloc(
            w->node_count * sizeof(arb_mat_struct));
    w->marginal_node_vectors = flint_malloc(
            w->node_count * sizeof(arb_mat_struct));
    for (i = 0; i < w->node_count; i++)
    {
        arb_mat_init(w->base_node_vectors+i, w->state_count, 1);
        arb_mat_init(w->lhood_node_vectors+i, w->state_count, 1);
        arb_mat_init(w->marginal_node_vectors+i, w->state_count, 1);
    }

    /* initialize per-edge state vectors */
    w->lhood_edge_vectors = flint_malloc(
            w->edge_count * sizeof(arb_mat_struct));
    for (i = 0; i < w->edge_count; i++)
    {
        arb_mat_init(w->lhood_edge_vectors+i, w->state_count, 1);
    }
}


static void
likelihood_ws_clear(likelihood_ws_t w)
{
    int i, idx;

    /* clear edge rates */
    _arb_vec_clear(w->edge_rates, w->edge_count);

    if (w->equilibrium)
    {
        _arb_vec_clear(w->equilibrium, w->state_count);
    }

    /* clear unscaled rate matrix */
    arb_mat_clear(w->rate_matrix);

    /* clear per-edge matrices */
    for (idx = 0; idx < w->edge_count; idx++)
    {
        arb_mat_clear(w->transition_matrices + idx);
        arb_mat_clear(w->lhood_edge_vectors + idx);
    }
    flint_free(w->transition_matrices);
    flint_free(w->lhood_edge_vectors);

    /* clear per-node matrices */
    for (i = 0; i < w->node_count; i++)
    {
        arb_mat_clear(w->base_node_vectors + i);
        arb_mat_clear(w->lhood_node_vectors + i);
        arb_mat_clear(w->marginal_node_vectors + i);
    }
    flint_free(w->base_node_vectors);
    flint_free(w->lhood_node_vectors);
    flint_free(w->marginal_node_vectors);
}


static void
likelihood_ws_update(likelihood_ws_t w, model_and_data_t m, slong prec)
{
    /* arrays are already allocated and initialized */
    int idx;
    arb_mat_struct *rmat, *tmat;

    rmat = w->rate_matrix;

    _update_rate_matrix_and_equilibrium(
            w->rate_matrix,
            w->equilibrium,
            m->rate_divisor,
            m->use_equilibrium_rate_divisor,
            m->root_prior,
            m->mat,
            prec);

    /* modify rate matrix diagonals so that the sum of each row is zero */
    _arb_update_rate_matrix_diagonal(rmat, prec);

    /* exponentiate scaled rate matrices */
    for (idx = 0; idx < w->edge_count; idx++)
    {
        tmat = w->transition_matrices + idx;
        arb_mat_scalar_mul_arb(tmat, rmat, w->edge_rates + idx, prec);
        arb_mat_exp(tmat, tmat, prec);
    }
}


static void
_nd_accum_update(nd_accum_t arr,
        likelihood_ws_t w, model_and_data_t m, slong prec)
{
    int site, i, j;
    arb_t lhood;
    arb_mat_struct *nvec;
    nd_axis_struct *site_axis, *node_axis, *state_axis;
    int site_count;
    int *coords;

    coords = malloc(arr->ndim * sizeof(int));

    arb_init(lhood);

    site_count = pmat_nsites(m->p);

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
                m->use_equilibrium_root_prior, w->equilibrium,
                m->preorder[0], prec);

        /*
         * Update per-node and per-edge likelihood vectors.
         * Actually the likelihood vectors on edges are not used.
         * This is a backward pass from the leaves to the root.
         */
        evaluate_site_lhood(lhood,
                w->lhood_node_vectors,
                w->lhood_edge_vectors,
                w->base_node_vectors,
                w->transition_matrices,
                m->g, m->preorder, w->node_count, prec);

        /*
         * Update marginal distribution vectors at nodes.
         * This is a forward pass from the root to the leaves.
         */
        evaluate_site_marginal(
                w->marginal_node_vectors,
                w->lhood_node_vectors,
                w->lhood_edge_vectors,
                w->transition_matrices,
                m->g, m->preorder, w->node_count, w->state_count, prec);

        /* update the nd accumulator */
        for (i = 0; i < w->node_count; i++)
        {
            /* skip nodes that are not requested */
            if (!node_axis->request_update[i]) continue;
            coords[1] = i;

            nvec = w->marginal_node_vectors + i;
            for (j = 0; j < w->state_count; j++)
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

    arb_clear(lhood);
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
    int ndim, result;
    int axis_idx;
    int site_count, node_count, state_count;
    nd_axis_struct axes[3];
    nd_accum_t arr;
    likelihood_ws_t w;

    result = 0;

    /* sites, nodes, states */
    ndim = 3;

    /* initialize counts */
    site_count = pmat_nsites(m->p);
    node_count = pmat_nrows(m->p);
    state_count = pmat_ncols(m->p);

    /* initialize likelihood workspace */
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
        /*
         * Update likelihood workspace.
         * This updates all members except the conditional and marginal
         * per-node and per-edge likelihood column state vectors.
         */
        likelihood_ws_update(w, m, prec);

        /* recompute axis reduction weights with increased precision */
        nd_axis_update_precision(axes+0, r_site, prec);
        nd_axis_update_precision(axes+1, r_node, prec);
        nd_axis_update_precision(axes+2, r_state, prec);

        /*
         * Recompute the output array with increased working precision.
         * This also updates the workspace conditional and merginal
         * per-node and per-edge likelihood column state vectors.
         */
        _nd_accum_update(arr, w, m, prec);

        /* check whether entries are accurate to full relative precision  */
        success = nd_accum_can_round(arr);
    }

    /* build the json output using the nd array */
    j_out = nd_accum_get_json(arr, &result);
    if (result) goto finish;

finish:

    /* clear likelihood workspace */
    likelihood_ws_clear(w);

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
        column_reduction_t r_node,
        column_reduction_t r_state, json_t *root)
{
    json_t *model_and_data = NULL;
    json_t *site_reduction = NULL;
    json_t *node_reduction = NULL;
    json_t *state_reduction = NULL;
    int site_count, node_count, state_count;
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
    site_count = pmat_nsites(m->p);
    node_count = pmat_nrows(m->p);
    state_count = pmat_ncols(m->p);

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
