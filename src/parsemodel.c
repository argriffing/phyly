/*
 * Module for loading a json file with a particular format.
 */

#include "stdlib.h"
#include "stdio.h"

#include "jansson.h"

#include "parsemodel.h"
#include "model.h"
#include "csr_graph.h"

/*
 * Array size must already be known.
 * Memory must already have been allocated.
 */
static int
_validate_nonnegative_array(double *dest, int desired_length, json_t *root)
{
    int i, n;

    if (!dest)
    {
        flint_fprintf(stderr, "_validate_nonnegative_array "
                "internal error: dest array is NULL\n");
        abort();
    }

    if (!json_is_array(root))
    {
        fprintf(stderr, "_validate_nonnegative_array: not an array\n");
        return -1;
    }

    n = json_array_size(root);

    if (n != desired_length)
    {
        fprintf(stderr, "_validate_nonnegative_array: unexpected array length "
                "(actual: %d desired: %d)\n", n, desired_length);
        return -1;
    }

    for (i = 0; i < desired_length; i++)
    {
        double d;
        json_t *x;

        x = json_array_get(root, i);
        if (!json_is_number(x))
        {
            fprintf(stderr, "_validate_nonnegative_array: not a number\n");
            return -1;
        }
        d = json_number_value(x);
        if (d < 0)
        {
            fprintf(stderr, "_validate_nonnegative_array: "
                    "array entries must be nonnegative\n");
            return -1;
        }
        dest[i] = d;
    }

    return 0;
}


/*
 * This optional argument can be a positive real number,
 * or it can be the string "equilibrium_exit_rate".
 */
static int
_validate_rate_divisor(model_and_data_t m, json_t *root)
{
    if (root && !json_is_null(root))
    {
        const char s_option[] = "equilibrium_exit_rate";
        const char s_msg[] = (
                "_validate_rate_divisor: the optional rate_divisor "
                "argument must be either a positive number or the string "
                "\"equilibrium_exit_rate\"\n");

        if (json_is_string(root))
        {
            if (!strcmp(json_string_value(root), s_option))
            {
                m->use_equilibrium_rate_divisor = 1;
            }
            else
            {
                fprintf(stderr, s_msg);
                return -1;
            }
        }
        else if (json_is_number(root))
        {
            double tmpd;
            tmpd = json_number_value(root);

            if (tmpd <= 0)
            {
                fprintf(stderr, s_msg);
                return -1;
            }

            arb_set_d(m->rate_divisor, tmpd);
        }
        else
        {
            fprintf(stderr, s_msg);
            return -1;
        }
    }

    return 0;
}


static int
_validate_root_prior(model_and_data_t m, json_t *root)
{
    int state_count = model_and_data_state_count(m);
    if (m->root_prior)
    {
        flint_fprintf(stderr, "_validate_root_prior "
                "internal error: root_prior is not NULL\n");
        abort();
    }
    if (root && !json_is_null(root))
    {
        const char s_equilibrium[] = "equilibrium_distribution";
        const char s_uniform[] = "uniform_distribution";
        const char s_msg[] = (
                "_validate_root_prior: the optional \"root_prior\" "
                "must be either a list of probabilities"
                "or one of the strings "
                "{\"equilibrium_distribution\", \"uniform_distribution\"\n");

        if (json_is_string(root))
        {
            if (!strcmp(json_string_value(root), s_equilibrium))
            {
                m->use_equilibrium_root_prior = 1;
            }
            else if (!strcmp(json_string_value(root), s_uniform))
            {
                m->use_uniform_root_prior = 1;
            }
            else
            {
                fprintf(stderr, s_msg);
                return -1;
            }
        }
        else
        {
            int result;
            m->root_prior = malloc(state_count * sizeof(double));
            result = _validate_nonnegative_array(
                    m->root_prior, state_count, root);
            if (result)
            {
                fprintf(stderr, s_msg);
                return result;
            }
        }
    }

    return 0;
}


static int
_validate_edge(int *pair, json_t *root)
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


static int
_validate_edges(model_and_data_t m, json_t *root)
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
        fprintf(stderr, "_validate_edges: not an array\n");
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
        result = _validate_edge(pair, edge);
        if (result) goto finish;

        /* require that the node indices are within a valid range */
        for (j = 0; j < 2; j++)
        {
            int idx;

            idx = pair[j];
            if (idx < 0 || idx >= node_count)
            {
                fprintf(stderr, "_validate_edges: node indices must be ");
                fprintf(stderr, "integers no less than 0 and no greater ");
                fprintf(stderr, "than the number of edges\n");
                result = -1;
                goto finish;
            }
        }

        /* require that edges have distinct endpoints */
        if (pair[0] == pair[1])
        {
            fprintf(stderr, "_validate_edges: edges cannot be loops\n");
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
            fprintf(stderr, "_validate_edges: exactly one node should have ");
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
            fprintf(stderr, "_validate_edges: the in-degree of each node ");
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
            fprintf(stderr, "_validate_edges: node index %d is not ", i);
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
            result = _validate_edge(pair, edge);
            if (result)
            {
                fprintf(stderr, "_validate_edges: internal error ");
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
            result = _validate_edge(pair, edge);
            if (result) goto finish;
            csr_edge_mapper_add_edge(m->edge_map, m->g, pair[0], pair[1]);
        }
    }

    /* define a topological ordering of the nodes */
    {
        if (m->preorder)
        {
            fprintf(stderr, "_validate_edges: expected NULL preorder\n");
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

static int
_validate_edge_rate_coefficients(model_and_data_t m, json_t *root)
{
    int edge_count, result;

    edge_count = m->g->nnz;
    m->edge_rate_coefficients = malloc(edge_count * sizeof(double));

    result = _validate_nonnegative_array(
            m->edge_rate_coefficients, edge_count, root);
    if (result)
    {
        fprintf(stderr, "_validate_edge_rate_coefficients: "
                "array validation has failed\n");
        return result;
    }

    return 0;
}

static int
_validate_rate_matrix(model_and_data_t m, json_t *root)
{
    int i, j;
    int n, col_count;
    json_t *x, *y;
    int result;
    double tmpd;

    result = 0;

    if (!json_is_array(root))
    {
        fprintf(stderr, "_validate_rate_matrix: not an array\n");
        result = -1;
        goto finish;
    }

    n = json_array_size(root);
    arb_mat_clear(m->mat);
    arb_mat_init(m->mat, n, n);

    for (i = 0; i < n; i++)
    {
        x = json_array_get(root, i);
        if (!json_is_array(x))
        {
            fprintf(stderr, "_validate_rate_matrix: ");
            fprintf(stderr, "this row is not an array\n");
            result = -1;
            goto finish;
        }
        col_count = json_array_size(x);
        if (col_count != n)
        {
            fprintf(stderr, "_validate_rate_matrix: this row length does not ");
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
                fprintf(stderr, "_validate_rate_matrix: not a number\n");
                result = -1;
                goto finish;
            }
            tmpd = json_number_value(y);
            if (tmpd < 0)
            {
                fprintf(stderr, "_validate_rate_matrix: ");
                fprintf(stderr, "rate matrix entries must be nonnegative\n");
                result = -1;
                goto finish;
            }
            arb_set_d(arb_mat_entry(m->mat, i, j), tmpd);
        }
    }

finish:

    return result;
}


static int
_validate_probability_array(model_and_data_t m, json_t *root)
{
    int node_count, state_count;
    int i, j, k;
    int site_count, n;
    json_t *x, *y, *z;
    int result;

    node_count = m->g->n;
    state_count = arb_mat_nrows(m->mat);

    result = 0;

    if (!json_is_array(root))
    {
        fprintf(stderr, "_validate_probability_array: expected an array\n");
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
            fprintf(stderr, "_validate_probability_array: expected an array\n");
            result = -1;
            goto finish;
        }

        n = json_array_size(x);
        if (n != node_count)
        {
            fprintf(stderr, "_validate_probability_array: failed to ");
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
                fprintf(stderr, "_validate_probability_array: ");
                fprintf(stderr, "expected an array\n");
                result = -1;
                goto finish;
            }

            n = json_array_size(y);
            if (n != state_count)
            {
                fprintf(stderr, "_validate_probability_array: failed to ");
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
                    fprintf(stderr, "_validate_probability_array: ");
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
    json_t *edges = NULL;
    json_t *edge_rate_coefficients = NULL;
    json_t *rate_matrix = NULL;
    json_t *probability_array = NULL;
    json_t *rate_divisor = NULL;
    json_t *root_prior = NULL;

    int result;
    json_error_t err;
    size_t flags;

    result = 0;
    flags = JSON_STRICT;

    result = json_unpack_ex(root, &err, flags,
            "{s:o, s:o, s:o, s:o, s?o, s?o}",
            "edges", &edges,
            "edge_rate_coefficients", &edge_rate_coefficients,
            "rate_matrix", &rate_matrix,
            "probability_array", &probability_array,
            "rate_divisor", &rate_divisor,
            "root_prior", &root_prior);
    if (result)
    {
        fprintf(stderr, "error: on line %d: %s\n", err.line, err.text);
        return result;
    }

    result = _validate_edges(m, edges);
    if (result) return result;

    result = _validate_edge_rate_coefficients(m, edge_rate_coefficients);
    if (result) return result;

    result = _validate_rate_matrix(m, rate_matrix);
    if (result) return result;

    result = _validate_probability_array(m, probability_array);
    if (result) return result;

    result = _validate_rate_divisor(m, rate_divisor);
    if (result) return result;

    result = _validate_root_prior(m, root_prior);
    if (result) return result;

    return result;
}
