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

typedef struct
{
    int *indices;
    int *indptr;
    int n;
    int nnz;
} csr_graph_struct;
typedef csr_graph_struct csr_graph_t[1];

void
csr_graph_init(csr_graph_t g)
{
    g->indices = NULL;
    g->indptr = NULL;
    g->n = 0;
    g->nnz = 0;
}

void
csr_graph_init_outdegree(csr_graph_t g, int n, int *out_degree)
{
    int i, nnz, accum;

    /* count the edges, and use this count to allocate the indices array  */
    accum = 0;
    for (i = 0; i < n; i++)
    {
        accum += out_degree[i];
    }
    nnz = accum;
    g->indices = malloc(nnz * sizeof(int));
    for (i = 0; i < nnz; i++)
    {
        g->indices[i] = -1;
    }

    /* define the indptr array */
    accum = 0;
    g->indptr = malloc((n+1) * sizeof(int));
    for (i = 0; i < n; i++)
    {
        g->indptr[i] = accum;
        accum += out_degree[i];
    }
    g->indptr[i] = accum;

    g->n = n;
    g->nnz = nnz;
}

void
csr_graph_clear(csr_graph_t g)
{
    free(g->indices);
    free(g->indptr);
    g->n = 0;
    g->nnz = 0;
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
validate_edges(int *node_count, int *edge_count, json_t *root)
{
    int root_node_index;
    int pair[2];
    int i, result;
    int *in_degree, *out_degree;
    int *preorder;
    json_t *edge;
    csr_graph_t g;

    csr_graph_init(g);

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

    *edge_count = json_array_size(root);
    *node_count = *edge_count + 1;
    in_degree = calloc(*node_count, sizeof(*in_degree));
    out_degree = calloc(*node_count, sizeof(*out_degree));

    for (i = 0; i < *edge_count; i++)
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
            if (idx < 0 || idx >= *node_count)
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
        for (i = 0; i < *node_count; i++)
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
    for (i = 0; i < *node_count; i++)
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
    for (i = 0; i < *node_count; i++)
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
        int *v;
        int n;
        n = *node_count;
        csr_graph_clear(g);
        csr_graph_init_outdegree(g, n, out_degree);
        v = calloc(n, sizeof(int));
        for (i = 0; i < *edge_count; i++)
        {
            int offset, a, b;
            edge = json_array_get(root, i);
            result = validate_edge(pair, edge);
            if (result)
            {
                fprintf(stderr, "validate_edges: internal error ");
                fprintf(stderr, "on second pass\n");
                abort();
            }
            a = pair[0];
            b = pair[1];
            offset = g->indptr[a] + v[a];
            g->indices[offset] = b;
            v[a]++;
        }
        free(v);
    }

    /* define a topological ordering of the nodes */
    {
        int n, nu, nv;
        int *visited, *u, *v, *tmp;
        int npre;
        int flag;
        flag = 0;
        n = *node_count;
        visited = calloc(n, sizeof(int));
        preorder = malloc(n * sizeof(int));
        u = malloc(n * sizeof(int));
        v = malloc(n * sizeof(int));
        npre = 0;
        nv = 1;
        v[0] = root_node_index;
        visited[0] = 1;
        while (!flag && nv)
        {
            int a, b, j;
            tmp = u; u = v; v = tmp;
            nu = nv;
            nv = 0;
            for (i = 0; !flag && i < nu; i++)
            {
                a = u[i];
                preorder[npre++] = a;
                for (j = g->indptr[a]; !flag && j < g->indptr[a+1]; j++)
                {
                    b = g->indices[j];
                    if (visited[b])
                    {
                        flag = 1;
                    }
                    else
                    {
                        v[nv++] = b;
                        visited[b] = 1;
                    }
                }
            }
        }
        free(u);
        free(v);
        free(visited);
        if (flag)
        {
            fprintf(stderr, "validate_edges: topo sort failed\n");
            result = -1;
            goto finish;
        }
        if (npre != n)
        {
            fprintf(stderr, "validate_edges: the topo sort contains ");
            fprintf(stderr, "%d of the %d nodes\n", npre, n);
            result = -1;
            goto finish;
        }
    }

finish:

    free(in_degree);
    free(out_degree);
    free(preorder);
    csr_graph_clear(g);
    return result;
}



int
validate_model_and_data(json_t *root)
{
    json_error_t err;
    int result;
    size_t flags;
    json_t *edges;
    json_t *edge_rate_coefficients;
    json_t *rate_matrix;
    json_t *probability_array;

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
        return result;
    }

    /* edges */
    {
        int node_count, edge_count;
        result = validate_edges(&node_count, &edge_count, edges);
        if (result) return result;
    }

    return 0;
}


json_t *arbplf_ll(void *userdata, json_t *root)
{
    int result;

    json_t *j_out = NULL;
    json_t *model_and_data = NULL;
    json_t *reductions = NULL;
    int working_precision = 0;
    const char *sum_product_strategy = NULL;

    if (userdata)
    {
        fprintf(stderr, "error: unexpected userdata\n");
        abort();
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
        int result;
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
            abort();
        }
    }

    /* validate the model and data section of the json input */
    result = validate_model_and_data(model_and_data);
    if (result) abort();

    /* create new json object */
    j_out = json_pack("{s:s}", "hello", "world");

    flint_cleanup();
    return j_out;
}
