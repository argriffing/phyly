#include "stdlib.h"

#include "arb.h"

#include "reduction.h"


void
column_reduction_init(column_reduction_t r)
{
    r->selection = NULL;
    r->weights = NULL;
    r->agg_mode = AGG_NONE;
    r->selection_len = 0;
}

void
column_reduction_clear(column_reduction_t r)
{
    free(r->selection);
    free(r->weights);
}

/*
 * Given the user-provided edge reduction,
 * define an edge aggregation weight vector
 * whose indices are with respect to csr tree edges.
 * Also define a weight divisor that applies to all weights.
 */
int
get_edge_agg_weights(
        arb_t weight_divisor, arb_struct * weights,
        int edge_count, int *order, column_reduction_t r, slong prec)
{
    int i, edge, idx;
    int * edge_selection_count = NULL;
    int result = 0;

    _arb_vec_zero(weights, edge_count);

    /* define edge selection count if necessary */
    if (r->agg_mode == AGG_SUM || r->agg_mode == AGG_AVG)
    {
        edge_selection_count = calloc(edge_count, sizeof(int));
        for (i = 0; i < r->selection_len; i++)
        {
            edge = r->selection[i];
            idx = order[edge];
            edge_selection_count[idx]++;
        }
    }

    if (r->agg_mode == AGG_WEIGHTED_SUM)
    {
        arb_t weight;
        arb_init(weight);
        for (i = 0; i < r->selection_len; i++)
        {
            edge = r->selection[i];
            arb_set_d(weight, r->weights[i]);
            idx = order[edge];
            arb_add(weights+idx, weights+idx, weight, prec);
        }
        arb_clear(weight);
        arb_one(weight_divisor);
    }
    else if (r->agg_mode == AGG_SUM)
    {
        for (idx = 0; idx < edge_count; idx++)
        {
            arb_set_si(weights+idx, edge_selection_count[idx]);
        }
        arb_one(weight_divisor);
    }
    else if (r->agg_mode == AGG_AVG)
    {
        for (idx = 0; idx < edge_count; idx++)
        {
            arb_set_si(weights+idx, edge_selection_count[idx]);
        }
        arb_set_si(weight_divisor, r->selection_len);
    }
    else if (r->agg_mode == AGG_ONLY)
    {
        if (r->selection_len != 1)
        {
            fprintf(stderr, "error: when using {\"aggregation\" : \"only\"}, "
                    "the selection length must be exactly 1\n");
            result = -1;
            goto finish;
        }
        arb_one(weight_divisor);
        arb_one(weights + 0);
    }
    else
    {
        fprintf(stderr, "internal error: unexpected aggregation mode\n");
        result = -1;
        goto finish;
    }

finish:

    free(edge_selection_count);
    return result;
}


int
get_column_agg_weights(
        arb_t weight_divisor, arb_struct * weights,
        int total_len, column_reduction_t r, slong prec)
{
    int i, idx;
    int * idx_selection_count = NULL;
    int result = 0;

    _arb_vec_zero(weights, total_len);

    /* define idx selection count if necessary */
    if (r->agg_mode == AGG_SUM || r->agg_mode == AGG_AVG)
    {
        idx_selection_count = calloc(total_len, sizeof(int));
        for (i = 0; i < r->selection_len; i++)
        {
            idx = r->selection[i];
            idx_selection_count[idx]++;
        }
    }

    if (r->agg_mode == AGG_WEIGHTED_SUM)
    {
        if (!r->weights) abort(); /* assert */
        if (prec)
        {
            arb_t tmp;
            arb_init(tmp);
            for (i = 0; i < r->selection_len; i++)
            {
                idx = r->selection[i];
                arb_set_d(tmp, r->weights[i]);
                /*
                flint_printf("debug: prec=%wd\n", prec);
                flint_printf("weight : ");
                arb_printd(weight, 15); flint_printf("\n");
                flint_printf("weights[idx] : ");
                arb_printd(weights + idx, 15); flint_printf("\n");
                */
                arb_add(weights+idx, weights+idx, tmp, prec);
            }
            arb_clear(tmp);
        }
        else
        {
            for (i = 0; i < r->selection_len; i++)
            {
                idx = r->selection[i];
                arb_indeterminate(weights + idx);
            }
        }
        arb_one(weight_divisor);
    }
    else if (r->agg_mode == AGG_SUM)
    {
        for (idx = 0; idx < total_len; idx++)
        {
            arb_set_si(weights+idx, idx_selection_count[idx]);
        }
        arb_one(weight_divisor);
    }
    else if (r->agg_mode == AGG_AVG)
    {
        for (idx = 0; idx < total_len; idx++)
        {
            arb_set_si(weights+idx, idx_selection_count[idx]);
        }
        arb_set_si(weight_divisor, r->selection_len);
    }
    else if (r->agg_mode == AGG_ONLY)
    {
        if (r->selection_len != 1)
        {
            fprintf(stderr, "error: when using {\"aggregation\" : \"only\"}, "
                    "the selection length must be exactly 1\n");
            result = -1;
            goto finish;
        }
        arb_one(weight_divisor);
        arb_one(weights + 0);
    }
    else
    {
        fprintf(stderr, "internal error: unexpected aggregation mode\n");
        result = -1;
        goto finish;
    }

finish:

    free(idx_selection_count);
    return result;
}


int
get_site_agg_weights(
        arb_t weight_divisor, arb_struct * weights,
        int site_count, column_reduction_t r, slong prec)
{
    return get_column_agg_weights(
            weight_divisor, weights, site_count, r, prec);
}

int
get_node_agg_weights(
        arb_t weight_divisor, arb_struct * weights,
        int node_count, column_reduction_t r, slong prec)
{
    return get_column_agg_weights(
            weight_divisor, weights, node_count, r, prec);
}

int
get_state_agg_weights(
        arb_t weight_divisor, arb_struct * weights,
        int state_count, column_reduction_t r, slong prec)
{
    return get_column_agg_weights(
            weight_divisor, weights, state_count, r, prec);
}
