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
