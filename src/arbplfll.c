/*
 * Use arbitrary precision matrix operations to compute a log likelihood.
 * The JSON format is used for both input and output.
 * Arbitrary precision is used only internally;
 * error bounds are not available for the input or the output,
 * but instead double precision floating point representation is used.
 * If the precision level used for intermediate calculations
 * is determined to be insufficient for reporting an "mpfr"-like
 * output (as indicated by the arb function arb_can_round_mpfr)
 * then an error is reported instead.
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
 * "reductions" : [
 * {
 *  "columns" : ["site"],
 *  "selection" : [a, b, c, ...], (optional)
 *  "aggregation" : {"sum" | "avg" | [a, b, c, ...]} (optional)
 * }], (optional)
 * "working_precision" : a, (optional)
 * "sum_product_strategy" : {"brute_force" | "dynamic_programming"} (optional)
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

#include "runjson.h"
#include "csr_graph.h"


typedef struct
{
    double *data;
    int r;
    int c;
} dmat_struct;
typedef dmat_struct dmat_t[1];

static __inline__ int
dmat_nrows(dmat_t mat)
{
    return mat->r;
}

static __inline__ int
dmat_ncols(dmat_t mat)
{
    return mat->c;
}

static __inline__ double *
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
    csr_graph_t g;
    dmat_t mat;
    int root_node_index;
    int *preorder;
    int *edge_rate_coefficients;
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
}

void
model_and_data_clear(model_and_data_t m)
{
    free(m->preorder);
    free(m->edge_rate_coefficients);
    csr_graph_clear(m->g);
    dmat_clear(m->mat);
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


int
validate_edge(int *v, json_t *root)
{
    json_error_t err;
    int result;
    size_t flags;

    flags = JSON_STRICT;
    result = json_unpack_ex(root, &err, flags, "[i, i]", v+0, v+1);
    if (result)
    {
        fprintf(stderr, "error: on line %d: %s\n", err.line, err.text);
        return result;
    }

    return 0;
}


int
validate_edges(csr_graph_t g, json_t *root)
{
    int root_node_index;
    int pair[2];
    int i, result;
    int *in_degree, *out_degree;
    int *preorder;
    int edge_count, node_count;
    json_t *edge;

    in_degree = NULL;
    out_degree = NULL;
    preorder = NULL;
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
    root_node_index = -1;
    {
        int root_count = 0;
        for (i = 0; i < node_count; i++)
        {
            if (!in_degree[i])
            {
                root_node_index = i;
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
        csr_graph_clear(g);
        csr_graph_init_outdegree(g, n, out_degree);
        csr_graph_start_adding_edges(g);
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
            csr_graph_add_edge(g, pair[0], pair[1]);
        }
        csr_graph_stop_adding_edges(g);
    }

    /* define a topological ordering of the nodes */
    {
        int n;
        n = node_count;
        preorder = malloc(n * sizeof(int));
        result = csr_graph_get_tree_topo_sort(preorder, g, root_node_index);
        if (result)
        {
            goto finish;
        }
    }

finish:

    free(in_degree);
    free(out_degree);
    free(preorder);
    return result;
}


int
validate_edge_rate_coefficients(csr_graph_t g, json_t *root)
{
    int i;
    int edge_count;
    json_t *x;

    int *rates;
    int result;

    rates = NULL;
    result = 0;

    if (!json_is_array(root))
    {
        fprintf(stderr, "validate_edge_rate_coefficients: not an array\n");
        result = -1;
        goto finish;
    }

    edge_count = json_array_size(root);
    if (edge_count != g->nnz)
    {
        fprintf(stderr, "validate_edge_rate_coefficients: ");
        fprintf(stderr, "unexpected array length ");
        fprintf(stderr, "(actual: %d desired: %d)\n", edge_count, g->nnz);
        result = -1;
        goto finish;
    }

    rates = malloc(edge_count * sizeof(double));
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
        rates[i] = json_number_value(x);
    }

finish:

    free(rates);
    return result;
}


int
validate_rate_matrix(json_t *root)
{
    int i, j;
    int n, col_count;
    json_t *x, *y;
    dmat_t mat;
    int result;

    result = 0;
    dmat_pre_init(mat);

    if (!json_is_array(root))
    {
        fprintf(stderr, "validate_rate_matrix: not an array\n");
        result = -1;
        goto finish;
    }

    n = json_array_size(root);
    dmat_init(mat, n, n);

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
            y = json_array_get(x, i);
            if (!json_is_number(y))
            {
                fprintf(stderr, "validate_rate_matrix: not a number\n");
                result = -1;
                goto finish;
            }
            *dmat_entry(mat, i, j) = json_number_value(y);
        }
    }

finish:

    dmat_clear(mat);
    return result;
}


int
validate_model_and_data(json_t *root)
{
    json_error_t err;
    json_t *edges;
    json_t *edge_rate_coefficients;
    json_t *rate_matrix;
    json_t *probability_array;

    model_and_data_t m;
    int result;
    size_t flags;

    model_and_data_init(m);
    result = 0;
    flags = JSON_STRICT;

    /*
 * "model_and_data" : {
 *  "edges" : [[a, b], [c, d], ...],                 (#edges, 2)
 *  "edge_rate_coefficients" : [a, b, ...],          (#edges, )
 *  "rate_matrix" : [[a, b, ...], [c, d, ...], ...], (#states, #states)
 *  "probability_array" : [...]                      (#sites, #nodes, #states)
 *  */
        
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
        goto finish;
    }

    /* edges */
    {
        result = validate_edges(m->g, edges);
        if (result) goto finish;
    }

    /* edge_rate_coefficients */
    {
        result = validate_edge_rate_coefficients(m->g, edge_rate_coefficients);
        if (result) goto finish;
    }

    /* rate_matrix */
    {
        result = validate_rate_matrix(rate_matrix);
        if (result) goto finish;
    }

finish:

    model_and_data_clear(m);
    return result;
}


json_t *arbplf_ll_run(void *userdata, json_t *root, int *retcode)
{
    json_t *j_out = NULL;
    json_t *model_and_data = NULL;
    json_t *reductions = NULL;
    int working_precision = 0;
    const char *sum_product_strategy = NULL;
    int result = 0;

    if (userdata)
    {
        fprintf(stderr, "error: unexpected userdata\n");
        result = -1;
        goto finish;
    }

    /*
 * input format:
 * {
 * "model_and_data" : {
 *  "edges" : [[a, b], [c, d], ...],                 (#edges, 2)
 *  "edge_rate_coefficients" : [a, b, ...],          (#edges, )
 *  "rate_matrix" : [[a, b, ...], [c, d, ...], ...], (#states, #states)
 *  "probability_array" : [...]                      (#sites, #nodes, #states)
 * },
 * "reductions" : [
 * {
 *  "columns" : ["site"],
 *  "selection" : [a, b, c, ...], (optional)
 *  "aggregation" : {"sum" | "avg" | [a, b, c, ...]} (optional)
 * }], (optional)
 * "working_precision" : a, (optional)
 * "sum_product_strategy" : {"brute_force" | "dynamic_programming"} (optional)
 * }
 * */

    /* parse the json input */
    {
        json_error_t err;
        size_t flags;
        
        flags = JSON_STRICT;
        result = json_unpack_ex(root, &err, flags,
                "{s:o, s?o, s?i, s?s}",
                "model_and_data", &model_and_data,
                "reductions", &reductions,
                "working_precision", &working_precision,
                "sum_product_strategy", &sum_product_strategy);
        if (result)
        {
            fprintf(stderr, "error: on line %d: %s\n", err.line, err.text);
            goto finish;
        }
    }

    /* validate the model and data section of the json input */
    result = validate_model_and_data(model_and_data);
    if (result)
    {
        goto finish;
    }

    /* create new json object */
    j_out = json_pack("{s:s}", "hello", "world");

finish:

    *retcode = result;

    flint_cleanup();
    return j_out;
}
