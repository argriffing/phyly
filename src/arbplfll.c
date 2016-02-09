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

#include "runjson.h"
#include "csr_graph.h"

#define AGG_NONE 0
#define AGG_AVG 1
#define AGG_SUM 2
#define AGG_WEIGHTED_SUM 3

int _can_round(arb_t x)
{
    return arb_can_round_arf(x, 53, ARF_RND_NEAR);
}


typedef struct
{
    double *data;
    int r;
    int c;
} dmat_struct;
typedef dmat_struct dmat_t[1];

int
dmat_nrows(dmat_t mat)
{
    return mat->r;
}

int
dmat_ncols(dmat_t mat)
{
    return mat->c;
}

double *
dmat_entry(dmat_t mat, int i, int j)
{
    return mat->data + i * mat->c + j;
}

void
dmat_pre_init(dmat_t mat)
{
    /* either init or clear may follow this call */
    mat->data = NULL;
    mat->r = 0;
    mat->c = 0;
}

void
dmat_init(dmat_t mat, int r, int c)
{
    mat->data = malloc(r*c*sizeof(double));
    mat->r = r;
    mat->c = c;
}

void
dmat_clear(dmat_t mat)
{
    free(mat->data);
}




typedef struct
{
    double *data;
    int s;
    int r;
    int c;
} pmat_struct;
typedef pmat_struct pmat_t[1];

static __inline__ int
pmat_nsites(pmat_t mat)
{
    return mat->s;
}

static __inline__ int
pmat_nrows(pmat_t mat)
{
    return mat->r;
}

static __inline__ int
pmat_ncols(pmat_t mat)
{
    return mat->c;
}

static __inline__ double *
pmat_entry(pmat_t mat, int i, int j, int k)
{
    return mat->data + i * mat->r * mat->c + j * mat->c + k;
}

void
pmat_pre_init(pmat_t mat)
{
    /* either init or clear may follow this call */
    mat->data = NULL;
    mat->s = 0;
    mat->r = 0;
    mat->c = 0;
}

void
pmat_init(pmat_t mat, int s, int r, int c)
{
    mat->data = malloc(s*r*c*sizeof(double));
    mat->s = s;
    mat->r = r;
    mat->c = c;
}

void
pmat_clear(pmat_t mat)
{
    free(mat->data);
}





/*
 * The object lifetime extends across precision levels and site evaluations.
 */
typedef struct
{
    csr_graph_t g;
    dmat_t mat;
    pmat_t p;
    int root_node_index;
    int *preorder;
    double *edge_rate_coefficients;
    csr_edge_mapper_t edge_map;
} model_and_data_struct;
typedef model_and_data_struct model_and_data_t[1];

void
model_and_data_init(model_and_data_t m)
{
    m->root_node_index = -1;
    m->preorder = NULL;
    m->edge_rate_coefficients = NULL;
    csr_graph_init(m->g);
    dmat_pre_init(m->mat);
    pmat_pre_init(m->p);
    csr_edge_mapper_pre_init(m->edge_map);
}

void
model_and_data_clear(model_and_data_t m)
{
    free(m->preorder);
    free(m->edge_rate_coefficients);
    csr_graph_clear(m->g);
    dmat_clear(m->mat);
    pmat_clear(m->p);
    csr_edge_mapper_clear(m->edge_map);
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
    arb_mat_struct *node_column_vectors;
} likelihood_ws_struct;
typedef likelihood_ws_struct likelihood_ws_t[1];

void
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
        w->node_column_vectors = NULL;
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
    w->node_column_vectors = flint_malloc(
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
        nmat = w->node_column_vectors + i;
        arb_mat_init(nmat, w->state_count, 1);
    }
}

void
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
    }
    flint_free(w->transition_matrices);
    
    for (i = 0; i < w->node_count; i++)
    {
        arb_mat_clear(w->node_column_vectors + i);
    }
    flint_free(w->node_column_vectors);
}




void _arb_mat_mul_entrywise(arb_mat_t c, arb_mat_t a, arb_mat_t b, slong prec);

void
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


void
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

void
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


void
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



int
validate_edge(int *pair, json_t *root)
{
    json_error_t err;
    int result;
    size_t flags;

    flags = JSON_STRICT;
    result = json_unpack_ex(root, &err, flags, "[i, i]", pair+0, pair+1);
    if (result)
    {
        fprintf(stderr, "error: on line %d: %s\n", err.line, err.text);
        return result;
    }

    return 0;
}


int
validate_edges(model_and_data_t m, json_t *root)
{
    int pair[2];
    int i, result;
    int *in_degree, *out_degree;
    int edge_count, node_count;
    json_t *edge;

    in_degree = NULL;
    out_degree = NULL;
    result = 0;

    if (!json_is_array(root))
    {
        fprintf(stderr, "validate_edges: not an array\n");
        result = -1;
        goto finish;
    }

    edge_count = json_array_size(root);
    node_count = edge_count + 1;
    in_degree = calloc(node_count, sizeof(*in_degree));
    out_degree = calloc(node_count, sizeof(*out_degree));

    for (i = 0; i < edge_count; i++)
    {
        int j;

        edge = json_array_get(root, i);

        /* require that the edge is a json array of two integers */
        result = validate_edge(pair, edge);
        if (result) goto finish;

        /* require that the node indices are within a valid range */
        for (j = 0; j < 2; j++)
        {
            int idx;

            idx = pair[j];
            if (idx < 0 || idx >= node_count)
            {
                fprintf(stderr, "validate_edges: node indices must be ");
                fprintf(stderr, "integers no less than 0 and no greater ");
                fprintf(stderr, "than the number of edges\n");
                result = -1;
                goto finish;
            }
        }

        /* require that edges have distinct endpoints */
        if (pair[0] == pair[1])
        {
            fprintf(stderr, "validate_edges: edges cannot be loops\n");
            result = -1;
            goto finish;
        }

        /* update vertex degrees */
        out_degree[pair[0]]++;
        in_degree[pair[1]]++;
    }

    /* require that only a single node has in-degree 0 */
    {
        int root_count = 0;
        for (i = 0; i < node_count; i++)
        {
            if (!in_degree[i])
            {
                m->root_node_index = i;
                root_count++;
            }
        }
        if (root_count != 1)
        {
            fprintf(stderr, "validate_edges: exactly one node should have ");
            fprintf(stderr, "in-degree 0\n");
            result = -1;
            goto finish;
        }
    }

    /* require that the in-degree of each node is 0 or 1 */
    for (i = 0; i < node_count; i++)
    {
        if (in_degree[i] < 0 || in_degree[i] > 1)
        {
            fprintf(stderr, "validate_edges: the in-degree of each node ");
            fprintf(stderr, "must be 0 or 1\n");
            result = -1;
            goto finish;
        }
    }

    /* require that each node is an endpoint of at least one edge */
    for (i = 0; i < node_count; i++)
    {
        int degree;
        degree = in_degree[i] + out_degree[i];
        if (degree < 1)
        {
            fprintf(stderr, "validate_edges: node index %d is not ", i);
            fprintf(stderr, "an endpoint of any edge\n");
            result = -1;
            goto finish;
        }
    }

    /* define the edges of the csr graph */
    {
        int n;
        n = node_count;
        csr_graph_clear(m->g);
        csr_graph_init_outdegree(m->g, n, out_degree);
        csr_graph_start_adding_edges(m->g);
        for (i = 0; i < edge_count; i++)
        {
            edge = json_array_get(root, i);
            result = validate_edge(pair, edge);
            if (result)
            {
                fprintf(stderr, "validate_edges: internal error ");
                fprintf(stderr, "on second pass\n");
                goto finish;
            }
            csr_graph_add_edge(m->g, pair[0], pair[1]);
        }
        csr_graph_stop_adding_edges(m->g);
    }

    /*
     * Define a map from json edge index to csr edge index.
     * This will be used later when associating edge rate coefficients
     * with edges of the tree.
     */
    {
        csr_edge_mapper_init(m->edge_map, m->g->n, m->g->nnz);
        for (i = 0; i < edge_count; i++)
        {
            edge = json_array_get(root, i);
            result = validate_edge(pair, edge);
            if (result) goto finish;
            csr_edge_mapper_add_edge(m->edge_map, m->g, pair[0], pair[1]);
        }
    }

    /* define a topological ordering of the nodes */
    {
        if (m->preorder)
        {
            fprintf(stderr, "validate_edges: expected NULL preorder\n");
            result = -1;
            goto finish;
        }
        m->preorder = malloc(node_count * sizeof(int));
        result = csr_graph_get_tree_topo_sort(
                m->preorder, m->g, m->root_node_index);
        if (result)
        {
            goto finish;
        }
    }

finish:

    free(in_degree);
    free(out_degree);
    return result;
}


int
validate_edge_rate_coefficients(model_and_data_t m, json_t *root)
{
    int i;
    int n;
    int edge_count;
    json_t *x;

    int result;
    double tmpd;

    edge_count = m->g->nnz;
    result = 0;

    if (!json_is_array(root))
    {
        fprintf(stderr, "validate_edge_rate_coefficients: not an array\n");
        result = -1;
        goto finish;
    }

    n = json_array_size(root);
    if (n != edge_count)
    {
        fprintf(stderr, "validate_edge_rate_coefficients: ");
        fprintf(stderr, "unexpected array length ");
        fprintf(stderr, "(actual: %d desired: %d)\n", n, edge_count);
        result = -1;
        goto finish;
    }

    m->edge_rate_coefficients = malloc(edge_count * sizeof(double));
    for (i = 0; i < edge_count; i++)
    {
        x = json_array_get(root, i);
        if (!json_is_number(x))
        {
            fprintf(stderr, "validate_edge_rate_coefficients: ");
            fprintf(stderr, "not a number\n");
            result = -1;
            goto finish;
        }
        tmpd = json_number_value(x);
        m->edge_rate_coefficients[i] = tmpd;
    }

finish:

    return result;
}


int
validate_rate_matrix(model_and_data_t m, json_t *root)
{
    int i, j;
    int n, col_count;
    json_t *x, *y;
    int result;
    double tmpd;

    result = 0;

    if (!json_is_array(root))
    {
        fprintf(stderr, "validate_rate_matrix: not an array\n");
        result = -1;
        goto finish;
    }

    n = json_array_size(root);
    dmat_init(m->mat, n, n);

    for (i = 0; i < n; i++)
    {
        x = json_array_get(root, i);
        if (!json_is_array(x))
        {
            fprintf(stderr, "validate_rate_matrix: this row is not an array\n");
            result = -1;
            goto finish;
        }
        col_count = json_array_size(x);
        if (col_count != n)
        {
            fprintf(stderr, "validate_rate_matrix: this row length does not ");
            fprintf(stderr, "match the number of rows: ");
            fprintf(stderr, "(actual: %d desired: %d)\n", col_count, n);
            result = -1;
            goto finish;
        }
        for (j = 0; j < n; j++)
        {
            y = json_array_get(x, j);
            if (!json_is_number(y))
            {
                fprintf(stderr, "validate_rate_matrix: not a number\n");
                result = -1;
                goto finish;
            }
            tmpd = json_number_value(y);
            *dmat_entry(m->mat, i, j) = tmpd;
        }
    }

finish:

    return result;
}


int
validate_probability_array(model_and_data_t m, json_t *root)
{
    int node_count, state_count;
    int i, j, k;
    int site_count, n;
    json_t *x, *y, *z;
    int result;

    node_count = m->g->n;
    state_count = dmat_nrows(m->mat);

    result = 0;

    if (!json_is_array(root))
    {
        fprintf(stderr, "validate_probability_array: ");
        fprintf(stderr, "validate_probability_array: expected an array\n");
        result = -1;
        goto finish;
    }

    site_count = json_array_size(root);
    pmat_init(m->p, site_count, node_count, state_count);

    for (i = 0; i < site_count; i++)
    {
        x = json_array_get(root, i);
        if (!json_is_array(x))
        {
            fprintf(stderr, "validate_probability_array: expected an array\n");
            result = -1;
            goto finish;
        }

        n = json_array_size(x);
        if (n != node_count)
        {
            fprintf(stderr, "validate_probability_array: failed to ");
            fprintf(stderr, "match the number of nodes: ");
            fprintf(stderr, "(actual: %d desired: %d)\n", n, node_count);
            result = -1;
            goto finish;
        }
        for (j = 0; j < node_count; j++)
        {
            y = json_array_get(x, j);
            if (!json_is_array(y))
            {
                fprintf(stderr, "validate_probability_array: ");
                fprintf(stderr, "expected an array\n");
                result = -1;
                goto finish;
            }

            n = json_array_size(y);
            if (n != state_count)
            {
                fprintf(stderr, "validate_probability_array: failed to ");
                fprintf(stderr, "match the number of states: ");
                fprintf(stderr, "(actual: %d desired: %d)\n", n, state_count);
                result = -1;
                goto finish;
            }
            for (k = 0; k < state_count; k++)
            {
                z = json_array_get(y, k);
                if (!json_is_number(z))
                {
                    fprintf(stderr, "validate_probability_array: ");
                    fprintf(stderr, "not a number\n");
                    result = -1;
                    goto finish;
                }
                *pmat_entry(m->p, i, j, k) = json_number_value(z);
            }
        }
    }

finish:

    return result;
}


int
validate_model_and_data(model_and_data_t m, json_t *root)
{
    json_t *edges;
    json_t *edge_rate_coefficients;
    json_t *rate_matrix;
    json_t *probability_array;

    int result;
    json_error_t err;
    size_t flags;

    result = 0;
    flags = JSON_STRICT;

    /* all four members are json objects and all are required */
    result = json_unpack_ex(root, &err, flags,
            "{s:o, s:o, s:o, s:o}",
            "edges", &edges,
            "edge_rate_coefficients", &edge_rate_coefficients,
            "rate_matrix", &rate_matrix,
            "probability_array", &probability_array);
    if (result)
    {
        fprintf(stderr, "error: on line %d: %s\n", err.line, err.text);
        return result;
    }

    result = validate_edges(m, edges);
    if (result) return result;

    result = validate_edge_rate_coefficients(m, edge_rate_coefficients);
    if (result) return result;

    result = validate_rate_matrix(m, rate_matrix);
    if (result) return result;

    result = validate_probability_array(m, probability_array);
    if (result) return result;

    return result;
}

typedef struct
{
    int *selection;
    int *weights;
    int agg_mode;
    int selection_len;
} site_reduction_struct;
typedef site_reduction_struct site_reduction_t[1];

void
site_reduction_init(site_reduction_t r)
{
    r->selection = NULL;
    r->weights = NULL;
    r->agg_mode = AGG_NONE;
    r->selection_len = 0;
}

void
site_reduction_clear(site_reduction_t r)
{
    free(r->selection);
    free(r->weights);
}

int
validate_site_selection(site_reduction_t r, int site_count, json_t *root)
{
    json_t *x;
    int i, site;
    if (!root)
    {
        r->selection_len = site_count;
        r->selection = malloc(site_count * sizeof(int));
        for (i = 0; i < site_count; i++)
        {
            r->selection[i] = i;
        }
    }
    else
    {
        if (!json_is_array(root))
        {
            fprintf(stderr, "validate_site_selection: not an array\n");
            return -1;
        }
        r->selection_len = json_array_size(root);
        r->selection = malloc(r->selection_len * sizeof(int));
        for (i = 0; i < r->selection_len; i++)
        {
            x = json_array_get(root, i);
            if (!json_is_integer(x))
            {
                fprintf(stderr, "validate_site_selection: ");
                fprintf(stderr, "the given site index is not an integer\n");
                return -1;
            }
            site = json_integer_value(x);
            if (site < 0)
            {
                fprintf(stderr, "validate_site_selection: ");
                fprintf(stderr, "site indices must be positive\n");
                return -1;
            }
            if (site >= site_count)
            {
                fprintf(stderr, "validate_site_selection: ");
                fprintf(stderr, "site indices must be less than ");
                fprintf(stderr, "the number of sites\n");
                return -1;
            }
            r->selection[i] = site;
        }
    }
    return 0;
}

int
validate_site_aggregation(site_reduction_t r, json_t *root)
{
    json_t *x;
    int i;
    double weight;
    int n;
    if (!root)
    {
        r->agg_mode = AGG_NONE;
    }
    else if (json_is_string(root))
    {
        if (!strcmp(json_string_value(root), "sum"))
        {
            r->agg_mode = AGG_SUM;
        }
        else if (!strcmp(json_string_value(root), "avg"))
        {
            r->agg_mode = AGG_AVG;
        }
        else
        {
            fprintf(stderr, "validate_site_aggregation: ");
            fprintf(stderr, "the only valid site aggregation strings are ");
            fprintf(stderr, "\"sum\" and \"avg\"\n");
            return -1;
        }
    }
    else if (json_is_array(root))
    {
        r->agg_mode = AGG_WEIGHTED_SUM;
        n = json_array_size(root);
        if (n != r->selection_len)
        {
            fprintf(stderr, "validate_site_aggregation: ");
            fprintf(stderr, "the number of weights must be equal to the ");
            fprintf(stderr, "number of selected sites, ");
            fprintf(stderr, "or to the total number of sites if no ");
            fprintf(stderr, "selection was provided\n");
            return -1;
        }
        r->weights = malloc(n * sizeof(double));
        for (i = 0; i < n; i++)
        {
            x = json_array_get(root, i);
            if (!json_is_number(x))
            {
                fprintf(stderr, "validate_site_selection: ");
                fprintf(stderr, "the given site index is not an integer\n");
                return -1;
            }
            weight = json_number_value(x);
            r->weights[i] = weight;
        }
    }
    else
    {
        fprintf(stderr, "validate_site_aggregation: ");
        fprintf(stderr, "the optional site aggregation should be ");
        fprintf(stderr, "either \"sum\" or \"avg\" or an array of weights\n");
        return -1;
    }
    return 0;
}


int
validate_site_reduction(site_reduction_t r, int site_count, json_t *root)
{
    json_t *selection = NULL;
    json_t *aggregation = NULL;

    int result;
    json_error_t err;
    size_t flags;

    result = 0;
    flags = JSON_STRICT;

    if (root)
    {
        result = json_unpack_ex(root, &err, flags,
                "{s?o, s?o}",
                "selection", &selection,
                "aggregation", &aggregation);
        if (result)
        {
            fprintf(stderr, "error: on line %d: %s\n", err.line, err.text);
            return result;
        }
    }

    result = validate_site_selection(r, site_count, selection);
    if (result) return result;

    result = validate_site_aggregation(r, aggregation);
    if (result) return result;

    return result;
}


void
evaluate_site_likelihood(
        arb_t lhood, model_and_data_t m, likelihood_ws_t w, int site)
{
    int u, a, b, idx;
    int start, stop;
    int state;
    double tmpd;
    arb_mat_struct * nmat;
    arb_mat_struct * nmatb;
    arb_mat_struct * tmat;
    csr_graph_struct *g;

    g = m->g;

    for (u = 0; u < w->node_count; u++)
    {
        a = m->preorder[w->node_count - 1 - u];
        nmat = w->node_column_vectors + a;
        start = g->indptr[a];
        stop = g->indptr[a+1];

        /* initialize the state vector for node a */
        for (state = 0; state < w->state_count; state++)
        {
            tmpd = *pmat_entry(m->p, site, a, state);
            arb_set_d(arb_mat_entry(nmat, state, 0), tmpd);
        }

        /* Hadamard-accumulate matrix-vector products. */
        for (idx = start; idx < stop; idx++)
        {
            b = g->indices[idx];

            tmat = w->transition_matrices + idx;
            nmatb = w->node_column_vectors + b;
            /*
             * At this point (a, b) is an edge from node a to node b
             * in a post-order traversal of edges of the tree.
             */
            _prune_update(nmat, nmat, tmat, nmatb, w->prec);
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


json_t *arbplf_ll_run(void *userdata, json_t *root, int *retcode)
{
    json_t *j_out = NULL;
    json_t *model_and_data = NULL;
    json_t *site_reduction = NULL;
    int result = 0;
    int site_count = 0;
    model_and_data_t m;
    site_reduction_t r;
    int *site_is_selected = NULL;
    arb_struct * site_likelihoods = NULL;
    arb_struct * site_log_likelihoods = NULL;
    arb_t aggregate, tmp, weight;
    likelihood_ws_t w;
    int iter = 0;

    arb_init(tmp);
    arb_init(weight);
    arb_init(aggregate);

    likelihood_ws_init(w, NULL, 0);
    model_and_data_init(m);
    site_reduction_init(r);

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

    site_count = pmat_nsites(m->p);

    /* validate the (optional) site reduction section of the json input */
    result = validate_site_reduction(r, site_count, site_reduction);
    if (result) goto finish;

    int i;
    int site;
    site_is_selected = calloc(site_count, sizeof(int));
    for (i = 0; i < r->selection_len; i++)
    {
        site = r->selection[i];
        site_is_selected[site] = 1;
    }

    site_likelihoods = _arb_vec_init(site_count);
    site_log_likelihoods = _arb_vec_init(site_count);

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
            ll = site_log_likelihoods + site;

            /* if the site is not in the selection then skip it */
            if (!site_is_selected[site])
            {
                continue;
            }

            /* if the log likelihood is fully evaluated then skip it */
            if (iter && r->agg_mode == AGG_NONE && _can_round(ll))
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
            if (r->agg_mode == AGG_NONE && !_can_round(ll))
            {
                failed = 1;
            }
        }
        /* compute the aggregate if any */
        if (r->agg_mode != AGG_NONE)
        {
            arb_zero(aggregate);
            arb_one(weight);
            for (i = 0; i < r->selection_len; i++)
            {
                site = r->selection[i];
                if (r->agg_mode == AGG_WEIGHTED_SUM)
                {
                    arb_set_d(weight, r->weights[i]);
                }
                ll = site_log_likelihoods + site;
                arb_addmul(aggregate, ll, weight, prec);
            }
            if (r->agg_mode == AGG_AVG)
            {
                arb_div_si(aggregate, aggregate, r->selection_len, prec);
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

    /* create the json object */
    /*
 * output format (without aggregation of the "site" column):
 * {
 *  "columns" : ["site", "value"],
 *  "data" : [[a, b], [c, d], ..., [y, z]] (# selected sites)
 * }
 *
 * */

    if (r->agg_mode == AGG_NONE)
    {
        double d;
        json_t *j_data, *x;
        j_data = json_array();
        for (i = 0; i < r->selection_len; i++)
        {
            site = r->selection[i];
            ll = site_log_likelihoods + site;
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
    if (site_log_likelihoods)
    {
        _arb_vec_clear(site_log_likelihoods, site_count);
    }
    site_reduction_clear(r);
    model_and_data_clear(m);
    likelihood_ws_clear(w);
    flint_cleanup();
    return j_out;
}
