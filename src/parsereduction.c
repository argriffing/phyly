#include "stdio.h"
#include "string.h"
#include "stdlib.h"

#include "jansson.h"

#include "parsereduction.h"
#include "reduction.h"


static int
_validate_column_selection(column_reduction_t r,
        int k, const char *name, json_t *root)
{
    json_t *x;
    int i, value;
    if (!root)
    {
        r->selection_len = k;
        r->selection = malloc(k * sizeof(int));
        for (i = 0; i < k; i++)
        {
            r->selection[i] = i;
        }
    }
    else
    {
        if (!json_is_array(root))
        {
            fprintf(stderr, "error: %s selection: ", name);
            fprintf(stderr, "the selection should be an array\n");
            return -1;
        }
        r->selection_len = json_array_size(root);
        r->selection = malloc(r->selection_len * sizeof(int));
        for (i = 0; i < r->selection_len; i++)
        {
            x = json_array_get(root, i);
            if (!json_is_integer(x))
            {
                fprintf(stderr, "error: %s selection: ", name);
                fprintf(stderr, "each index in the selection "
                        "must be an integer\n");
                return -1;
            }
            value = json_integer_value(x);
            if (value < 0)
            {
                fprintf(stderr, "error: %s selection: ", name);
                fprintf(stderr, "each index in the selection "
                        "must be non-negative\n");
                return -1;
            }
            if (value >= k)
            {
                fprintf(stderr, "error: %s selection: ", name);
                fprintf(stderr, "each index in the selection "
                        "must be less than the total number of "
                        "available %s indices\n", name);
                return -1;
            }
            r->selection[i] = value;
        }
    }
    return 0;
}


static int
_validate_column_aggregation(column_reduction_t r,
        const char *name, json_t *root)
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
            fprintf(stderr, "error: %s aggregation (string): ", name);
            fprintf(stderr, "the only valid aggregation strings are "
                    "\"sum\" and \"avg\"\n");
            return -1;
        }
    }
    else if (json_is_array(root))
    {
        r->agg_mode = AGG_WEIGHTED_SUM;
        n = json_array_size(root);
        if (n != r->selection_len)
        {
            fprintf(stderr, "error: %s aggregation (weighted sum): ", name);
            fprintf(stderr, "the number of weights must be equal to "
                    "the number of selected %s indices, "
                    "or to the total number of %s indices "
                    "if no selection was provided\n", name, name);
            return -1;
        }
        r->weights = malloc(n * sizeof(double));
        for (i = 0; i < n; i++)
        {
            x = json_array_get(root, i);
            if (!json_is_number(x))
            {
                fprintf(stderr, "error: %s aggregation (weighted sum): ", name);
                fprintf(stderr, "weights should be numeric\n");
                return -1;
            }
            weight = json_number_value(x);
            r->weights[i] = weight;
        }
    }
    else
    {
        fprintf(stderr, "error: %s aggregation: ", name);
        fprintf(stderr, "if provided, the aggregation should be either "
                "the string \"sum\", or the string \"avg\", "
                "or an array of numeric weights for a weighted sum\n");
        return -1;
    }
    return 0;
}


int
validate_column_reduction(column_reduction_t r,
        int k, const char *name, json_t *root)
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

    result = _validate_column_selection(r, k, name, selection);
    if (result) return result;

    result = _validate_column_aggregation(r, name, aggregation);
    if (result) return result;

    return result;
}



/*
 * Column pair reduction.
 * This can be applied to transition reductions on state pairs, for example.
 */


static int
_validate_column_pair(int *pair, json_t *root)
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
_validate_column_pair_selection(column_reduction_t r,
        int **first_idx, int **second_idx,
        int k0, int k1, const char *name, json_t *root)
{
    json_t *j_pair;
    int i;
    int pair[2];
    int idx;
    int result = 0;

    if (!json_is_array(root))
    {
        fprintf(stderr, "error: %s selection: ", name);
        fprintf(stderr, "the selection should be an array\n");
        result = -1;
        goto finish;
    }
    r->selection_len = json_array_size(root);
    r->selection = malloc(r->selection_len * sizeof(int));
    *first_idx = flint_malloc(r->selection_len * sizeof(int));
    *second_idx = flint_malloc(r->selection_len * sizeof(int));
    for (i = 0; i < r->selection_len; i++)
    {
        j_pair = json_array_get(root, i);

        /* require that the pair is a json array of two integers */
        result = _validate_column_pair(pair, j_pair);
        if (result) goto finish;

        /* require that the indices are within a valid range */
        idx = pair[0];
        if (idx < 0 || idx >= k0)
        {
            fprintf(stderr, "error: %s selection: ", name);
            fprintf(stderr, "indices should be nonnegative integers less than "
                            "the number of column elements\n");
            result = -1;
            goto finish;
        }

        /* require that the indices are within a valid range */
        idx = pair[1];
        if (idx < 0 || idx >= k1)
        {
            fprintf(stderr, "error: %s selection: ", name);
            fprintf(stderr, "indices should be nonnegative integers less than "
                            "the number of column elements\n");
            result = -1;
            goto finish;
        }

        /* update the indices */
        (*first_idx)[i] = pair[0];
        (*second_idx)[i] = pair[1];

        /* the pair selection will simply be the index */
        r->selection[i] = i;
    }

finish:

    return result;
}


int
validate_column_pair_reduction(column_reduction_t r,
        int **first_idx, int **second_idx,
        int k0, int k1, const char *name, json_t *root)
{
    json_t *selection = NULL;
    json_t *aggregation = NULL;

    int result;
    json_error_t err;
    size_t flags;

    result = 0;
    flags = JSON_STRICT;

    *first_idx = NULL;
    *second_idx = NULL;

    if (root)
    {
        /* For pair reduction, 'selection' is required (not optional). */
        result = json_unpack_ex(root, &err, flags,
                "{s:o, s?o}",
                "selection", &selection,
                "aggregation", &aggregation);
        if (result)
        {
            fprintf(stderr, "error: on line %d: %s\n", err.line, err.text);
            return result;
        }
    }

    result = _validate_column_pair_selection(r, first_idx, second_idx,
            k0, k1, name, selection);
    if (result) return result;

    result = _validate_column_aggregation(r, name, aggregation);
    if (result) return result;

    return result;
}
