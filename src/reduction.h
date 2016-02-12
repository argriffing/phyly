#ifndef REDUCTION_H
#define REDUCTION_H

#include "arb.h"

#define AGG_NONE 0
#define AGG_AVG 1
#define AGG_SUM 2
#define AGG_WEIGHTED_SUM 3

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    int *selection;
    double *weights;
    int agg_mode;
    int selection_len;
} column_reduction_struct;
typedef column_reduction_struct column_reduction_t[1];

void column_reduction_init(column_reduction_t r);
void column_reduction_clear(column_reduction_t r);

int get_edge_agg_weights(
        arb_t weight_divisor, arb_struct * weights,
        int edge_count, int *order, column_reduction_t r, slong prec);

int get_site_agg_weights(
        arb_t weight_divisor, arb_struct * weights,
        int site_count, column_reduction_t r, slong prec);

#ifdef __cplusplus
}
#endif

#endif
