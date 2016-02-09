#include "stdlib.h"

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
