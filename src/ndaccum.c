#include "ndaccum.h"
#include "util.h"

#define DEBUG_ROUNDING 0

void
nd_axis_update_precision(nd_axis_t axis, column_reduction_t r, slong prec)
{
    int result;
    if (axis->agg_weights)
    {
        result = get_column_agg_weights(
                axis->agg_weight_divisor, axis->agg_weights, axis->n, r, prec);
        if (result) abort();
    }
}

void
nd_axis_init(nd_axis_t axis,
        const char *name, int total_count, column_reduction_t r,
        int component_axis_count, struct nd_component_axis *component_axes,
        slong prec)
{
    int i, idx;

    /* set compound axis hack data */
    axis->component_axis_count = component_axis_count;
    axis->component_axes = component_axes;

    /* set the name */
    axis->name = malloc(strlen(name) + 1);
    strcpy(axis->name, name);

    /* set the counts */
    axis->n = total_count;
    axis->k = r->selection_len;

    /* allocate some bookkeeping vectors  */
    axis->selection = calloc(axis->k, sizeof(int));
    axis->is_selected = calloc(axis->n, sizeof(int));
    axis->request_update = calloc(axis->n, sizeof(int));

    /* initialize the bookkeeping vectors */
    for (i = 0; i < axis->k; i++)
    {
        idx = r->selection[i];
        axis->selection[i] = idx;
        axis->is_selected[idx] = 1;
        axis->request_update[idx] = 1;
    }

    /* initialize the arbitrary precision weight arrays */
    if (r->agg_mode == AGG_NONE)
    {
        axis->agg_weights = NULL;
    }
    else
    {
        axis->agg_weights = _arb_vec_init(axis->n);
        arb_init(axis->agg_weight_divisor);
    }

    /* initialize weight values */
    nd_axis_update_precision(axis, r, prec);
}

void
nd_axis_clear(nd_axis_t axis)
{
    free(axis->name);
    free(axis->selection);
    free(axis->is_selected);
    free(axis->request_update);

    if (axis->agg_weights)
    {
        _arb_vec_clear(axis->agg_weights, axis->n);
        arb_clear(axis->agg_weight_divisor);
    }

    axis->n = 0;
    axis->k = 0;
}




void
nd_accum_pre_init(nd_accum_t a)
{
    a->ndim = 0;
    a->size = 0;
    a->shape = NULL;
    a->strides = NULL;
    a->axes = NULL;
    a->data = NULL;
}

void
nd_accum_init(nd_accum_t a, nd_axis_struct *axes, int ndim)
{
    int i;
    int stride;
    nd_axis_struct *axis;

    a->ndim = ndim;
    a->axes = axes;

    /* determine the nd array shape, accounting for aggregation along axes  */
    a->shape = malloc(a->ndim * sizeof(int));
    for (i = 0; i < ndim; i++)
    {
        axis = a->axes + i;
        if (axis->agg_weights)
        {
            a->shape[i] = 1;
        }
        else
        {
            a->shape[i] = axis->n;
        }
    }

    /* determine the nd array size, accounting for aggregation along axes */
    a->size = 1;
    for (i = 0; i < a->ndim; i++)
    {
        a->size *= a->shape[i];
    }

    /* determine the nd array strides, accounting for aggregation along axes */
    stride = 1;
    a->strides = malloc(a->ndim * sizeof(int));
    for (i = ndim-1; i >= 0; i--)
    {
        a->strides[i] = stride;
        stride *= a->shape[i];
    }

    /* allocate the data array */
    a->data = _arb_vec_init(a->size);
}

void
nd_accum_printd(const nd_accum_t a, slong digits)
{
    slong i;

    /* print structural data */
    flint_printf("ndim=%d\n", a->ndim);
    flint_printf("size=%d\n", a->size);

    flint_printf("shape=");
    for (i = 0; i < a->ndim; i++)
    {
        flint_printf("%d ", a->shape[i]);
    }
    flint_printf("\n");

    flint_printf("debug: strides=");
    for (i = 0; i < a->ndim; i++)
    {
        flint_printf("%d ", a->strides[i]);
    }
    flint_printf("\n");
}

/* todo: do this more intelligently and update axis requests for precision */
int
nd_accum_can_round(nd_accum_t a)
{
    int i;
    for (i = 0; i < a->size; i++)
    {
        if (!_can_round(a->data + i))
        {
            if (DEBUG_ROUNDING)
            {
                flint_printf("debug: failed to round element %d of %d:\n",
                        i, a->size);
                arb_printd(a->data + i, 15); flint_printf("\n");
            }
            return 0;
        }
    }
    return 1;
}

void
nd_accum_clear(nd_accum_t a)
{
    free(a->shape);
    free(a->strides);
    _arb_vec_clear(a->data, a->size);
}

void
nd_accum_accumulate(nd_accum_t a, int *coords, arb_struct *value, slong prec)
{
    int axis_idx;
    int offset, coord, stride;
    nd_axis_struct *axis;
    arb_struct *p;
    arb_t x;

    arb_init(x);
    arb_set(x, value);
    offset = 0;
    for (axis_idx = 0; axis_idx < a->ndim; axis_idx++)
    {
        axis = a->axes + axis_idx;
        coord = coords[axis_idx];
        stride = a->strides[axis_idx];

        /*
        flint_printf("debug:\n");
        flint_printf("ndim=%wd axis=%d coord=%d\n", a->ndim, axis_idx, coord);
        flint_printf("strides:\n");
        {
            int j;
            for (j = 0; j < a->ndim; j++)
            {
                flint_printf("%d : %d\n", j, a->strides[j]);
            }
        }
        */

        if (axis->agg_weights)
        {
            arb_mul(x, x, axis->agg_weights + coord, prec);

            /* todo: delay this division */
            arb_div(x, x, axis->agg_weight_divisor, prec);
        }
        else
        {
            offset += coord * stride;
        }
    }
    if (offset < 0)
    {
        fprintf(stderr, "internal error: negative offset\n");
        abort();
    }
    if (offset >= a->size)
    {
        fprintf(stderr, "internal error: offset=%d >= size=%d\n",
                offset, a->size);
        abort();
    }
    p = a->data + offset;
    arb_add(p, p, x, prec);
    arb_clear(x);
}

static void
_nd_accum_recursively_zero_requested_cells(nd_accum_t a,
        int axis_idx, int offset)
{
    int i, next_offset;
    nd_axis_struct *axis;

    /* define the current axis */
    axis = a->axes + axis_idx;

    /* terminate recursion */
    if (axis_idx == a->ndim)
    {
        arb_zero(a->data + offset);
        return;
    }

    if (axis->agg_weights)
    {
        /* aggregated axes do not rule out any cells */
        next_offset = offset;
        _nd_accum_recursively_zero_requested_cells(
                a, axis_idx+1, next_offset);
    }
    else
    {
        for (i = 0; i < axis->n; i++)
        {
            if (axis->request_update[i])
            {
                next_offset = offset + i * a->strides[axis_idx];
                _nd_accum_recursively_zero_requested_cells(
                        a, axis_idx+1, next_offset);
            }
        }
    }
}

void
nd_accum_zero_requested_cells(nd_accum_t a)
{
    _nd_accum_recursively_zero_requested_cells(a, 0, 0);
}

static int
_nd_accum_recursively_build_json(nd_accum_t a,
        json_t *j_rows, json_t *j_row, int axis_idx, int offset)
{
    int i, j, idx, next_offset;
    json_t *x;
    json_t *j_row_next;
    nd_axis_struct *axis;
    int result;

    result = 0;

    /* define the current axis */
    axis = a->axes + axis_idx;

    /* terminate recursion */
    if (axis_idx == a->ndim)
    {
        double d;
        d = _arb_get_d(a->data + offset);
        if (j_row)
        {
            j_row_next = json_deep_copy(j_row);
            x = json_real(d);
            json_array_append_new(j_row_next, x);
        }
        else
        {
            j_row_next = json_pack("[f]", d);
        }
        json_array_append_new(j_rows, j_row_next);
        return result;
    }

    if (axis->agg_weights)
    {
        /* skip aggregated axes */
        next_offset = offset;
        result = _nd_accum_recursively_build_json(
                a, j_rows, j_row, axis_idx+1, next_offset);
    }
    else
    {
        /* add selections to the row, and update the offset */
        for (i = 0; i < axis->k; i++)
        {
            idx = axis->selection[i];
            if (j_row)
            {
                j_row_next = json_deep_copy(j_row);
            }
            else
            {
                j_row_next = json_array();
            }
            if (axis->component_axes != NULL)
            {
                for (j = 0; j < axis->component_axis_count; j++)
                {
                    int component_idx;
                    component_idx = axis->component_axes[j].indices[i];
                    x = json_integer(component_idx);
                    json_array_append_new(j_row_next, x);
                }
            }
            else
            {
                x = json_integer(idx);
                json_array_append_new(j_row_next, x);
            }
            next_offset = offset + idx * a->strides[axis_idx];
            result = _nd_accum_recursively_build_json(
                    a, j_rows, j_row_next, axis_idx+1, next_offset);
            if (result) return result;
            json_decref(j_row_next);
        }
    }
    return result;
}


json_t *
nd_accum_get_json(nd_accum_t a, int *result_out)
{
    int j, axis_idx;
    int offset;
    nd_axis_struct *axis;
    json_t *j_out, *j_headers, *j_rows;
    int result;

    result = 0;

    /* build column header list */
    j_headers = json_array();
    {
        json_t *j_header;
        for (axis_idx = 0; axis_idx < a->ndim; axis_idx++)
        {
            axis = a->axes+axis_idx;
            if (!axis->agg_weights)
            {
                if (axis->component_axes != NULL)
                {
                    for (j = 0; j < axis->component_axis_count; j++)
                    {
                        j_header = json_string(axis->component_axes[j].name);
                        json_array_append_new(j_headers, j_header);
                    }
                }
                else
                {
                    j_header = json_string(axis->name);
                    json_array_append_new(j_headers, j_header);
                }
            }
        }
        j_header = json_string("value");
        json_array_append_new(j_headers, j_header);
    }

    /* recursively build the data array */
    {
        json_t *j_row;
        j_rows = json_array();
        j_row = NULL;
        axis_idx = 0;
        offset = 0;
        result = _nd_accum_recursively_build_json(
                a, j_rows, j_row, axis_idx, offset);
    }

    *result_out = result;

    j_out = json_pack("{s:o, s:o}",
        "columns", j_headers,
        "data", j_rows);
    return j_out;
}
