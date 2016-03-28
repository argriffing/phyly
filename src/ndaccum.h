#ifndef NDACCUM_H
#define NDACCUM_H

/*
 * Orthogonal axis reduction of an n-dimensional array.
 */

#include "jansson.h"
#include "arb.h"

#include "reduction.h"


#ifdef __cplusplus
extern "C" {
#endif


struct nd_component_axis
{
    const char *name;
    int *indices; /* length = k = n of nd_axis_struct */
};


typedef struct
{
    char *name;

    /* a hack to allow 'compound' nd axes; these are not owned
     * by this struct */
    int component_axis_count;
    struct nd_component_axis *component_axes;

    /* the total number of indices along this axis */
    int n;

    /* if not aggregating, the length of the selection */
    int k;

    /*
     * Selection for output, may be larger or smaller than n.
     * Mutually exclusive with agg_weights.
     */
    int *selection; /* length k */

    /* indicates whether or not each index is selected */
    int *is_selected; /* length n */

    /* indicates whether or not each index requires updated precision */
    int *request_update; /* length n */

    /*
     * NULL or indicates the aggregation weight for each axis.
     * This already accounts for multiple selections of the same index.
     */
    arb_struct *agg_weights; /* length n */
    arb_t agg_weight_divisor;

} nd_axis_struct;
typedef nd_axis_struct nd_axis_t[1];

void nd_axis_update_precision(nd_axis_t axis, column_reduction_t r, slong prec);
void nd_axis_init(nd_axis_t axis,
        const char *name, int total_count, column_reduction_t r,
        int component_axis_count, struct nd_component_axis *component_axes,
        slong prec);
void nd_axis_clear(nd_axis_t axis);


/* the axis array is not owned by this object */
typedef struct
{
    int ndim;
    int size;
    int *shape;
    int *strides;
    nd_axis_struct *axes;
    arb_struct *data;
} nd_accum_struct;
typedef nd_accum_struct nd_accum_t[1];

void nd_accum_pre_init(nd_accum_t a);
void nd_accum_init(nd_accum_t a, nd_axis_struct *axes, int ndim);
int nd_accum_can_round(nd_accum_t a);
void nd_accum_clear(nd_accum_t a);
void nd_accum_accumulate(nd_accum_t a,
        int *coords, arb_struct *value, slong prec);
void nd_accum_zero_requested_cells(nd_accum_t a);
json_t * nd_accum_get_json(nd_accum_t a, int *result_out);
void nd_accum_printd(const nd_accum_t a, slong digits);



#ifdef __cplusplus
}
#endif

#endif
