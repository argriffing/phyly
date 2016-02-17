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

#include "parsemodel.h"
#include "parsereduction.h"
#include "runjson.h"
#include "arbplfmarginal.h"


typedef struct
{
    char *name;

    /* the total number of indices along this axis */
    int n;

    /* if not aggregating, the length of the selection */
    int k;

    /*
     * Selection for output, may be larger or smaller than n.
     * Mutually exclusive with agg_weights.
     */
    int *selection; /* length k */

    /* indicates whether or not each index is selected */
    int *is_selected; /* length n */

    /* indicates whether or not each index requires updated precision */
    int *request_update; /* length n */

    /*
     * NULL or indicates the aggregation weight for each axis.
     * This already accounts for multiple selections of the same index.
     */
    arb_struct *agg_weights; /* length n */
    arb_t agg_weight_divisor;

} nd_axis_struct;
typedef nd_axis_struct nd_axis_t[1];


static void
nd_axis_update_precision(nd_axis_t axis, column_reduction_t r, slong prec)
{
    result = get_column_agg_weights(
            axis->agg_weight_divisor, axis->agg_weights, axis->n, r, prec);
    if (result) abort();
}

static void
nd_axis_init(nd_axis_t axis,
        const char *name, int total_count, column_reduction_t r, slong prec)
{
    int i, idx, result;

    /* set the name */
    axis->name = malloc(strlen(name) + 1);
    strcpy(axis->name, name);

    /* set the counts */
    axis->n = total_count;
    axis->k = r->selection_len;

    /* allocate some bookkeeping vectors  */
    axis->selection = malloc(axis->k * sizeof(int));
    axis->is_selected = malloc(axis->n * sizeof(int));
    axis->request_update = malloc(axis->n * sizeof(int));

    /* initialize the bookkeeping vectors */
    for (i = 0; i < axis->n; i++)
    {
        axis->is_selected[i] = 0;
    }
    for (i = 0; i < axis->k; i++)
    {
        idx = r->selection[i];
        axis->selection[i] = idx;
        axis->is_selected[idx] = 1;
        axis->request_update[idx] = 1;
    }

    /* initialize the arbitrary precision weight arrays */
    arb_init(axis->agg_weight_divisor);
    axis->agg_weights = arb_vec_init(axis->n);

    /* initialize weight values */
    nd_axis_update_precision(axis, r, prec);
}

static void
nd_axis_clear(nd_axis_t axis)
{
    free(axis->name);
    free(axis->selection);
    free(axis->is_selected);
    free(axis->request_update);

    _arb_vec_clear(axis->agg_weights, axis->n);
    arb_clear(axis->agg_divisor);

    axis->n = 0;
    axis->k = 0;
}



/* the axis array is not owned by this object */
typedef struct
{
    int ndim;
    int size;
    int *shape;
    int *strides;
    nd_axis_struct *axes;
    arb_struct *data;
} nd_accum_struct;
typedef nd_accum_struct nd_accum_t[1];

static void
nd_accum_pre_init(nd_accum_t a)
{
    a->ndim = 0;
    a->size = 0;
    a->shape = NULL;
    a->strides = NULL;
    a->axes = NULL;
    a->data = NULL;
}

static void
nd_accum_init(nd_accum_t a, nd_axis_struct *axes, int ndim)
{
    int i;
    int stride;

    a->ndim = ndim;
    a->axes = axes;

    /* determine the nd array shape, accounting for aggregation along axes  */
    a->shape = malloc(a->ndim * sizeof(int));
    for (i = 0; i < ndim; i++)
    {
        if (a->axes[i]->agg_weights)
        {
            a->shape[i] = 1;
        }
        else
        {
            a->shape[i] = a->axes[i]->n;
        }
    }

    /* determine the nd array size, accounting for aggregation along axes */
    a->size = 1;
    for (i = 0; i < a->ndim; i++)
    {
        a->size *= a->shape[i];
    }

    /* determine the nd array strides, accounting for aggregation along axes */
    stride = 1;
    a->shape = malloc(a->ndim * sizeof(int));
    for (i = ndim-1; i >= 0; i--)
    {
        a->strides[i] = stride;
        stride *= a->shape[i];
    }

    /* allocate the data array */
    a->data = _arb_vec_init(a->data, a->size);
}

/* todo: do this more intelligently and update axis requests for precision */
static int
nd_accum_can_round(nd_accum_t a)
{
    int i;
    for (i = 0; i < a->size; i++)
    {
        if (!_can_round(a->data + i))
        {
            return 0;
        }
    }
    return 1;
}

static void
nd_accum_clear(nd_accum_t a)
{
    free(a->shape);
    free(a->strides);
    _arb_vec_clear(a->data, a->size);
}

static void
nd_accum_accumulate(nd_accum_t a, int *coords, arb_struct *value, slong prec)
{
    int axis_idx;
    int i;
    int offset, coord, stride;
    nd_axis_struct *axis;
    arb_struct *p;
    arb_t x;

    arb_init_set(x, value);
    offset = 0;
    for (axis_idx = 0; axis_idx < a->ndim; axis_idx++)
    {
        coord = coords[axis_idx];
        axis = a->axes[axis_idx];
        stride = a->strides[axis_idx];
        if (axis->agg_weights)
        {
            arb_mul(x, x, axis->agg_weights[coord], prec);

            /* todo: delay this division */
            arb_div(x, x, axis->agg_weight_divisor, prec);
        }
        offset += coord * stride;
    }
    p = a->data + offset;
    arb_add(p, p, x, prec);
    arb_clear(x);
}

static void
_nd_accum_recursively_zero_requested_cells(nd_accum_t a,
        int axis_idx, int offset)
{
    int i, next_offset;
    nd_axis_struct *axis;

    /* define the current axis */
    axis = a->axes[axis_idx];

    /* terminate recursion */
    if (axis_idx == a->ndim)
    {
        arb_zero(a->data + offset);
        return result;
    }

    if (axis->agg_weights)
    {
        /* aggregated axes do not rule out any cells */
        next_offset = offset;
        _nd_accum_recursively_zero_requested_cells(
                a, axis_idx+1, next_offset);
    }
    else
    {
        for (i = 0; i < axis->n; i++)
        {
            if (axis->request_update[i])
            {
                next_offset = offset + i * a->strides[axis_idx];
                _nd_accum_recursively_zero_requested_cells(
                        a, axis_idx+1, next_offset);
            }
        }
    }
}

static void
nd_accum_zero_requested_cells(nd_accum_t a)
{
    _nd_accum_recursively_zero_requested_cells(a, 0, 0);
}

static int
_nd_accum_recursively_build_json(nd_accum_t a,
        json_t *j_rows, json_t *j_row, int axis_idx, int offset)
{
    int i, idx, next_offset;
    json_t *x;
    json_t *j_row_next;
    nd_axis_struct *axis;
    int result;

    result = 0;

    /* define the current axis */
    axis = a->axes[axis_idx];

    /* terminate recursion */
    if (axis_idx == a->ndim)
    {
        double d;
        d = arf_get_d(arb_midref(a->data + offset), ARF_RND_NEAR);
        if (j_row)
        {
            j_row_next = json_deep_copy(j_row);
            x = json_real(d);
            json_array_append_new(j_row_next, x);
        }
        else
        {
            j_row_next = json_pack("[f]", d);
        }
        json_array_append_new(j_data, j_row);
        return result;
    }

    if (axis->agg_weights)
    {
        /* skip aggregated axes */
        next_offset = offset;
        j_row_next = j_row;
        result = _nd_accum_recursively_build_json(
                a, j_rows, j_row, axis_idx+1, next_offset);
    }
    else
    {
        /* add selections to the row, and update the offset */
        if (j_row)
        {
            j_row_next = json_deep_copy(j_row);
        }
        else
        {
            j_row_next = json_array();
        }
        for (i = 0; i < axis->k; i++)
        {
            idx = axis->selection[i];
            x = json_integer(idx);
            json_array_append_new(j_row_next, x);
            next_offset = offset + idx * a->strides[axis_idx];
            result = _nd_accum_recursively_build_json(
                    a, j_rows, j_row_next, axis_idx+1, next_offset);
            if (result) return result;
        }
    }
    return result;
}


static json_t *
nd_accum_get_json(nd_accum_t a, int *result_out)
{
    int axis_idx;
    double d;
    int idx;
    nd_axis_struct *axis;
    json_t *j_data, *j_headers, *j_header, *x;
    int result;

    /* build column header list */
    j_headers = json_array();
    for (axis_idx = 0; axis_idx < a->ndim; axis_idx++)
    {
        axis = axes[axis_idx];
        if (axis->agg_weights)
        {
            j_header = json_string(axis->name);
            json_array_append_new(j_data, j_header);
        }
    }
    j_header = json_string("value");
    json_array_append_new(j_header);

    /* recursively build the data array */
    j_rows = json_array();
    j_row = NULL;
    axis_idx = 0;
    offset = 0;
    result = _nd_accum_recursively_build_json(
            a, j_rows, j_row, axis_idx, offset);
    *result_out = result;

    j_out = json_pack("{s:o, s:o}",
        "columns", j_headers,
        "data", j_data);
    return j_out;
}


/* Likelihood workspace. */
typedef struct
{
    int node_count;
    int edge_count;
    int state_count;
    arb_struct *edge_rates;
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
    int i, n;
    arb_mat_struct *tmat;
    double tmpd;

    g = m->g;

    w->node_count = g->n;
    w->edge_count = g->nnz;
    w->state_count = arb_mat_nrows(m->mat);

    w->edge_rates = _arb_vec_init(w->edge_count);
    arb_mat_init(w->rate_matrix, w->state_count, w->state_count);
    w->transition_matrices = flint_malloc(
            w->edge_count * sizeof(arb_mat_struct));

    /* intialize transition probability matrices */
    for (idx = 0; idx < w->edge_count; idx++)
    {
        tmat = w->transition_matrices + idx;
        arb_mat_init(tmat, w->state_count, w->state_count);
    }

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

    /* modify rate matrix diagonals so that the sum of each row is zero */
    dmat_get_arb_mat(rmat, m->mat);
    _arb_update_rate_matrix_diagonal(rmat, prec);

    /* exponentiate scaled rate matrices */
    for (idx = 0; idx < w->edge_count; idx++)
    {
        tmat = w->transition_matrices + idx;
        arb_mat_scalar_mul_arb(tmat, rmat, w->edge_rates + idx, prec);
        arb_mat_exp(tmat, tmat, prec);
    }
}


/* Calculate the likelihood, storing many intermediate calculations. */
static void
evaluate_site_likelihood(arb_t lhood,
        arb_struct *lhood_node_vectors,
        arb_struct *lhood_edge_vectors,
        arb_struct *base_node_vectors,
        arb_struct *transition_matrices,
        csr_graph_struct *g, int *preorder, int node_count, slong prec)
{
    int u, a, b, idx;
    int start, stop;
    arb_mat_struct *nmat, *nmatb, *tmat, *emat;

    /*
     * Fill all of the per-node and per-edge likelihood-related vectors.
     * Note that because edge derivatives are requested,
     * the vectors on edges are stored explicitly.
     * In the likelihood-only variant of this function, these per-edge
     * vectors are temporary variables whose lifespan is only long enough
     * to update the vector associated with the parent node of the edge.
     */
    for (u = 0; u < node_count; u++)
    {
        a = preorder[node_count - 1 - u];
        nmat = lhood_node_vectors + a;
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
            tmat = transition_matrices + idx;
            nmatb = lhood_node_vectors + b;
            emat = lhood_edge_vectors + idx;
            arb_mat_mul(emat, tmat, nmatb, prec);
        }

        /* initialize the state vector for node a */
        arb_mat_set(nmat, base_node_vectors + a);

        /* multiplicatively accumulate state vectors at this node */
        for (idx = start; idx < stop; idx++)
        {
            emat = lhood_edge_vectors + idx;
            _arb_mat_mul_entrywise(nmat, nmat, emat, prec);
        }
    }

    /* Report the sum of state entries associated with the root. */
    int root_node_index = preorder[0];
    nmat = lhood_node_vectors + root_node_index;
    _arb_mat_sum(lhood, nmat, prec);
}


static void
evaluate_marginal_distributions(
        arb_mat_struct *marginal_node_vectors,
        arb_mat_struct *lhood_node_vectors,
        arb_mat_struct *transition_matrices,
        csr_graph *g, int *preorder,
        int node_count, int state_count, slong prec)
{
    int u, a;
    int start, stop;
    arb_mat_struct *lvec, *mvec, *mvecb;
    arb_mat_t rvec;
    arb_t s;

    arb_mat_init(rvec, 1, state_count);
    arb_init(s);

    _arb_mat_ones(marginal_node_vectors + preorder[0]);

    for (u = 0; u < node_count; u++)
    {
        a = preorder[u];
        lvec = lhood_node_vectors + a;
        mvec = marginal_node_vectors + a;
        start = g->indptr[a];
        stop = g->indptr[a+1];

        /*
         * Entrywise multiply by the likelihood node vector
         * and then normalize the distribution.
         */
        _arb_mat_mul_entrywise(mvec, mvec, lvec, prec);
        _arb_mat_sum(s, mvec, prec);
        arb_mat_scalar_div_arb(mvec, mvec, s, prec);

        /* initialize neighboring downstream marginal vectors */
        for (idx = start; idx < stop; idx++)
        {
            b = g->indices[idx];
            /*
             * At this point (a, b) is an edge from node a to node b
             * in a pre-order traversal of edges of the tree.
             */
            mvecb = marginal_node_vectors + b;
            arb_mat_transpose(rvec, mvec);
            tmat = transition_matrices + idx;
            arb_mat_mul(rvec, rvec, tmat, prec);
            arb_mat_transpose(mvecb, rvec);

            /* todo: rewrite to avoid explicit transposes */
        }
    }

    arb_mat_clear(rvec);
    arb_clear(s);
}


static void
_nd_accum_update(nd_accum_t arr,
        likelihood_ws_t w, model_and_data_t m, slong prec)
{
    int site, i, j;
    arb_t lhood;
    arb_mat_struct *bvec, *nvec;
    nd_axis_struct *site_axis, *node_axis, *state_axis;
    int *coords;

    coords = malloc(arr->ndim * sizeof(int));

    arb_init(lhood);

    site_axis = arr->axes[0];
    node_axis = arr->axes[1];
    state_axis = arr->axes[2];

    /* zero all requested cells of the array */
    nd_accum_zero_requested_cells(arr);

    /*
     * Update the output array at the given precision.
     * Axes have already been updated for this precision.
     * The nd array links to axis selection and aggregation information,
     * and the model and data are provided separately.
     */
    for (site = 0; site < w->site_count; site++)
    {
        /* skip sites that are not requested */
        if (!site_axis->request_update[site]) continue;
        coords[0] = site;

        /* update base node vectors */
        for (i = 0; i < w->node_count; i++)
        {
            bvec = w->base_node_vectors + i;
            for (j = 0; j < w->state_count; j++)
            {
                arb_set_d(
                        arb_mat_entry(bvec, j, 0),
                        *pmat_entry(m->p, site, i, j));
            }
        }

        /*
         * Update per-node and per-edge likelihood vectors.
         * Actually the likelihood vectors on edges are not used.
         * This is a backward pass from the leaves to the root.
         */
        evaluate_site_likelihood(lhood,
                w->lhood_node_vectors,
                w->lhood_edge_vectors,
                w->base_node_vectors,
                w->transition_matrices,
                m->g, m->preorder, w->node_count, prec);

        /*
         * Update marginal distribution vectors at nodes.
         * This is a forward pass from the root to the leaves.
         */
        evaluate_marginal_distributions(
                w->marginal_node_vectors,
                w->lhood_node_vectors,
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
    int i, ndim, result;
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
    nd_axis_init(axes+0, "site", site_count, site_r, 0);
    nd_axis_init(axes+1, "node", node_count, node_r, 0);
    nd_axis_init(axes+2, "state", state_count, state_r, 0);

    /* initialize nd accumulation array */
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
        nd_axis_update_precision(axes+0, site_r, prec);
        nd_axis_update_precision(axes+1, node_r, prec);
        nd_axis_update_precision(axes+2, state_r, prec);

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
        nd_axis_clear(axes+i);
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
