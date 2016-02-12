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


int get_site_agg_weights(
        arb_t weight_divisor, arb_struct * weights,
        int site_count, column_reduction_t r, slong prec)
{
    int i, site;
    int * site_selection_count = NULL;
    int result = 0;

    _arb_vec_zero(weights, site_count);

    /* define site selection count if necessary */
    if (r->agg_mode == AGG_SUM || r->agg_mode == AGG_AVG)
    {
        site_selection_count = calloc(site_count, sizeof(int));
        for (i = 0; i < r->selection_len; i++)
        {
            site = r->selection[i];
            site_selection_count[site]++;
        }
    }

    if (r->agg_mode == AGG_WEIGHTED_SUM)
    {
        arb_t weight;
        arb_init(weight);
        for (i = 0; i < r->selection_len; i++)
        {
            site = r->selection[i];
            arb_set_d(weight, r->weights[i]);
            arb_add(weights+site, weights+site, weight, prec);
        }
        arb_clear(weight);
        arb_one(weight_divisor);
    }
    else if (r->agg_mode == AGG_SUM)
    {
        for (site = 0; site < site_count; site++)
        {
            arb_set_si(weights+site, site_selection_count[site]);
        }
        arb_one(weight_divisor);
    }
    else if (r->agg_mode == AGG_AVG)
    {
        for (site = 0; site < site_count; site++)
        {
            arb_set_si(weights+site, site_selection_count[site]);
        }
        arb_set_si(weight_divisor, r->selection_len);
    }
    else
    {
        fprintf(stderr, "internal error: unexpected aggregation mode\n");
        result = -1;
        goto finish;
    }

finish:

    free(site_selection_count);
    return result;
}
