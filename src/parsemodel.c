/*
 * Module for loading a json file with a particular format.
 */

#include "stdlib.h"
#include "stdio.h"

#include "jansson.h"

#include "parsemodel.h"
#include "model.h"
#include "csr_graph.h"
#include "rate_mixture.h"

/* helper function */
static int
_exists(const json_t *value)
{
    return (value && !json_is_null(value));
}

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
    if (m->root_prior->mode != ROOT_PRIOR_UNDEFINED)
    {
        flint_fprintf(stderr, "_validate_root_prior "
                "internal error: root prior is already defined\n");
        abort();
    }
    if (!root || json_is_null(root))
    {
        root_prior_init(m->root_prior, state_count, ROOT_PRIOR_NONE);
    }
    else
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
            enum root_prior_mode mode;
            if (!strcmp(json_string_value(root), s_equilibrium))
            {
                mode = ROOT_PRIOR_EQUILIBRIUM;
            }
            else if (!strcmp(json_string_value(root), s_uniform))
            {
                mode = ROOT_PRIOR_UNIFORM;
            }
            else
            {
                fprintf(stderr, s_msg);
                return -1;
            }
            root_prior_init(m->root_prior, state_count, mode);
        }
        else
        {
            int result;
            root_prior_init(m->root_prior, state_count, ROOT_PRIOR_CUSTOM);
            m->root_prior->custom_distribution = malloc(
                    state_count * sizeof(double));
            result = _validate_nonnegative_array(
                    m->root_prior->custom_distribution, state_count, root);
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

    /* initialize the navigation object */
    result = navigation_init(m->navigation, m->g,
            node_count, edge_count, m->root_node_index);
    if (result) goto finish;

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
    int i, j;
    int site_count, n;
    json_t *x, *y;
    const char name[] = "_validate_probability_array";
    int result = 0;
    int node_count = m->g->n;
    int state_count = arb_mat_nrows(m->mat);

    if (!json_is_array(root))
    {
        fprintf(stderr, "%s: expected an array\n", name);
        result = -1; goto finish;
    }

    site_count = json_array_size(root);
    pmat_init(m->p, site_count, node_count, state_count);

    for (i = 0; i < site_count; i++)
    {
        x = json_array_get(root, i);
        if (!json_is_array(x))
        {
            fprintf(stderr, "%s: expected an array\n", name);
            result = -1; goto finish;
        }

        n = json_array_size(x);
        if (n != node_count)
        {
            fprintf(stderr, "%s: failed to match the number of nodes: "
                    "(actual: %d desired: %d)\n", name, n, node_count);
            result = -1; goto finish;
        }
        for (j = 0; j < node_count; j++)
        {
            y = json_array_get(x, j);
            result = _validate_nonnegative_array(
                    pmat_entry(m->p, i, j, 0), state_count, y);
            if (result)
            {
                fprintf(stderr, "%s: array validation has failed\n", name);
                goto finish;
            }
        }
    }

finish:

    return result;
}


/* this initializes the probability array */
static int
_validate_character_data_and_definitions(model_and_data_t m,
        json_t *character_data, json_t *character_definitions)
{
    int i, j, k;
    int site_count, character_count, n;
    json_t *x, *y;
    const char name[] = "_validate_character_data_and_definitions";
    int node_count = m->g->n;
    int state_count = arb_mat_nrows(m->mat);
    int result = 0;
    double *defn = NULL;

    if (!_exists(character_data)) abort(); /* assert */

    if (!_exists(character_definitions))
    {
        fprintf(stderr, "%s: 'character_data' has been provided without "
                "'character_definitions'\n", name);
    }

    if (!json_is_array(character_data))
    {
        fprintf(stderr, "%s: expected 'character_data' "
                "to be an array\n", name);
        result = -1; goto finish;
    }

    if (!json_is_array(character_definitions))
    {
        fprintf(stderr, "%s: expected 'character_definitions' "
                "to be an array\n", name);
        result = -1; goto finish;
    }

    site_count = json_array_size(character_data);
    pmat_init(m->p, site_count, node_count, state_count);

    character_count = json_array_size(character_definitions);
    defn = flint_malloc(character_count * state_count * sizeof(double));

    /* validate the character definitions */
    for (j = 0; j < character_count; j++)
    {
        y = json_array_get(character_definitions, j);
        result = _validate_nonnegative_array(
                defn + j*state_count, state_count, y);
        if (result)
        {
            fprintf(stderr, "%s: validation of definition of "
                    "character %d of %d has failed\n",
                    name, j, character_count);
            goto finish;
        }
    }

    /* validate the character data */
    for (i = 0; i < site_count; i++)
    {
        x = json_array_get(character_data, i);
        if (!json_is_array(x))
        {
            fprintf(stderr, "%s: expected an array\n", name);
            result = -1; goto finish;
        }

        n = json_array_size(x);
        if (n != node_count)
        {
            fprintf(stderr, "%s: failed to match the number of nodes: "
                    "(actual: %d desired: %d)\n", name, n, node_count);
            result = -1; goto finish;
        }
        for (j = 0; j < node_count; j++)
        {
            int character_idx;

            y = json_array_get(x, j);
            if (!json_is_integer(y))
            {
                fprintf(stderr, "%s: character indices "
                        "must be integers\n", name);
                result = -1; goto finish;
            }

            character_idx = json_integer_value(y);
            if (character_idx < 0)
            {
                fprintf(stderr, "%s: character indices "
                        "must be non-negative\n", name);
                result = -1; goto finish;
            }

            if (character_idx >= character_count)
            {
                fprintf(stderr, "%s: character indices "
                        "must each be less than the character count (%d)\n",
                        name, character_count);
                result = -1; goto finish;
            }

            for (k = 0; k < state_count; k++)
            {
                int defn_offset = character_idx * state_count + k;
                *pmat_entry(m->p, i, j, k) = defn[defn_offset];
            }
        }
    }

finish:

    flint_free(defn);

    return result;
}


static int
_validate_rate_mixture(model_and_data_t m, json_t *root)
{
    int n;
    json_t *rates = NULL;
    json_t *prior = NULL;
    custom_rate_mixture_struct *cmix;

    int result = 0;
    json_error_t err;
    size_t flags = JSON_STRICT;
    rate_mixture_struct *x = m->rate_mixture;

    cmix = flint_malloc(sizeof(custom_rate_mixture_struct));
    x->custom_mix = cmix;
    result = json_unpack_ex(root, &err, flags,
            "{s:o, s:o}",
            "rates", &rates,
            "prior", &prior);
    if (result)
    {
        fprintf(stderr, "error: on line %d: %s\n", err.line, err.text);
        return result;
    }

    /* read the 'rates' array */
    {
        if (!json_is_array(rates))
        {
            fprintf(stderr, "_validate_rate_mixture: "
                    "'rates' is not an array\n");
            return -1;
        }

        n = json_array_size(rates);

        custom_rate_mixture_init(cmix, n);

        result = _validate_nonnegative_array(cmix->rates, n, rates);
        if (result)
        {
            fprintf(stderr, "_validate_rate_mixture: "
                    "invalid 'rates' array\n");
            return -1;
        }
    }

    /* read the 'prior' array (or the string "uniform_distribution") */
    {
        const char s_uniform[] = "uniform_distribution";
        const char s_msg[] = (
                "_validate_rate_mixture: the 'prior'"
                "argument must be either a nonnegative array or the string "
                "\"uniform_distribution\"\n");
        if (json_is_string(prior))
        {
            if (!strcmp(json_string_value(prior), s_uniform))
            {
                x->mode = RATE_MIXTURE_UNIFORM;
                cmix->mode = x->mode;
            }
            else
            {
                fprintf(stderr, s_msg);
                return -1;
            }
        }
        else if (json_is_array(prior))
        {
            result = _validate_nonnegative_array(cmix->prior, n, prior);
            if (result)
            {
                fprintf(stderr, "_validate_rate_mixture: "
                        "invalid 'prior' array\n");
                return -1;
            }
            x->mode = RATE_MIXTURE_CUSTOM;
            cmix->mode = x->mode;
        }
    }

    return result;
}

static int
_validate_gamma_rate_mixture(model_and_data_t m, json_t *root)
{
    json_t *gamma_shape = NULL;
    json_t *gamma_categories = NULL;
    json_t *invariable_prior = NULL;
    gamma_rate_mixture_struct *g;

    int result = 0;
    json_error_t err;
    size_t flags = JSON_STRICT;
    rate_mixture_struct *x = m->rate_mixture;

    x->mode = RATE_MIXTURE_GAMMA;

    g = flint_malloc(sizeof(custom_rate_mixture_struct));
    gamma_rate_mixture_init(g);
    x->gamma_mix = g;
    result = json_unpack_ex(root, &err, flags,
            "{s:o, s:o, s?o}",
            "gamma_shape", &gamma_shape,
            "gamma_categories", &gamma_categories,
            "invariable_prior", &invariable_prior);
    if (result)
    {
        fprintf(stderr, "error: on line %d: %s\n", err.line, err.text);
        return result;
    }

    /* read the invariable prior */
    if (!invariable_prior || json_is_null(invariable_prior))
    {
        g->invariable_prior = 0;
    }
    else if (json_is_number(invariable_prior))
    {
        g->invariable_prior = json_number_value(invariable_prior);
    }
    else
    {
        fprintf(stderr, "invariable_prior: not a number\n");
        result = -1;
        return result;
    }

    /* read the gamma shape */
    if (json_is_number(gamma_shape))
    {
        g->gamma_shape = json_number_value(gamma_shape);
    }
    else
    {
        fprintf(stderr, "gamma_shape: not a number\n");
        result = -1;
        return result;
    }

    /* read the number of gamma categories */
    if (json_is_integer(gamma_categories))
    {
        g->gamma_categories = json_integer_value(gamma_categories);
    }
    else
    {
        fprintf(stderr, "gamma_categories: not an integer\n");
        result = -1;
        return result;
    }

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
    json_t *rate_mixture = NULL;
    json_t *gamma_rate_mixture = NULL;
    json_t *character_definitions = NULL;
    json_t *character_data = NULL;

    int result;
    json_error_t err;
    size_t flags;

    result = 0;
    flags = JSON_STRICT;

    result = json_unpack_ex(root, &err, flags,
            "{s:o, s:o, s:o, s?o, s?o, s?o, s?o, s?o, s?o, s?o}",
            "edges", &edges,
            "edge_rate_coefficients", &edge_rate_coefficients,
            "rate_matrix", &rate_matrix,
            "probability_array", &probability_array,
            "character_definitions", &character_definitions,
            "character_data", &character_data,
            "rate_divisor", &rate_divisor,
            "root_prior", &root_prior,
            "rate_mixture", &rate_mixture,
            "gamma_rate_mixture", &gamma_rate_mixture);
    if (result)
    {
        fprintf(stderr, "error: on line %d: %s\n", err.line, err.text);
        return result;
    }

    /* check mutual exclusivity constraints */
    if (_exists(rate_mixture) && _exists(gamma_rate_mixture))
    {
        fprintf(stderr, "error: the mutually exclusive options "
                "'gamma_rate_mixture' and 'rate_mixture' "
                "have both been specified\n");
        return -1;
    }
    if (_exists(probability_array) && _exists(character_data))
    {
        fprintf(stderr, "error: the mutually exclusive options "
                "'probability_array' and 'character_data' "
                "have both been specified\n");
        return -1;
    }
    if (_exists(probability_array) && _exists(character_definitions))
    {
        fprintf(stderr, "error: the mutually exclusive options "
                "'probability_array' and 'character_definitions' "
                "have both been specified\n");
        return -1;
    }

    result = _validate_edges(m, edges);
    if (result) return result;

    result = _validate_edge_rate_coefficients(m, edge_rate_coefficients);
    if (result) return result;

    result = _validate_rate_matrix(m, rate_matrix);
    if (result) return result;

    if (_exists(probability_array))
    {
        result = _validate_probability_array(m, probability_array);
        if (result) return result;
    }
    else if (_exists(character_data))
    {
        result = _validate_character_data_and_definitions(m,
                character_data, character_definitions);
        if (result) return result;
    }
    else
    {
        fprintf(stderr, "error: either "
                "'probability_array' or 'character_data' "
                "must be specified\n");
        return -1;
    }

    result = _validate_rate_divisor(m, rate_divisor);
    if (result) return result;

    result = _validate_root_prior(m, root_prior);
    if (result) return result;

    if (_exists(gamma_rate_mixture))
    {
        result = _validate_gamma_rate_mixture(m, gamma_rate_mixture);
        if (result) return result;
    }
    else if (_exists(rate_mixture))
    {
        result = _validate_rate_mixture(m, rate_mixture);
        if (result) return result;
    }
    else
    {
        m->rate_mixture->mode = RATE_MIXTURE_NONE;
    }

    return result;
}
