#include "flint/flint.h"

#include "arb.h"

#include "util.h"
#include "arb_vec_extras.h"

void
_arb_vec_proportions(arb_struct *b, const arb_struct *a, slong n, slong prec)
{
    int i, zero_count, nonzero_count, indeterminate_count;
    arb_t total;
    arb_init(total);
    zero_count = 0;
    nonzero_count = 0;
    indeterminate_count = 0;
    for (i = 0; i < n; i++)
    {
        if (_arb_is_indeterminate(a + i))
            indeterminate_count++;
        else if (arb_is_zero(a + i))
            zero_count++;
        else if (!arb_contains_zero(a + i))
            nonzero_count++;
    }
    if (indeterminate_count || zero_count == n)
    {
        _arb_vec_indeterminate(b, n);
    }
    else if (nonzero_count == 1 && zero_count == n - 1)
    {
        for (i = 0; i < n; i++)
        {
            if (arb_is_zero(a + i))
                arb_zero(b + i);
            else
                arb_one(b + i);
        }
    }
    else
    {
        for (i = 0; i < n; i++)
            arb_add(total, total, a+i, prec);
        for (i = 0; i < n; i++)
            arb_div(b+i, a+i, total, prec);
    }
    arb_clear(total);
}

void
_arb_vec_div(arb_struct *c,
        const arb_struct *a, const arb_struct *b, slong n, slong prec)
{
    slong i;
    for (i = 0; i < n; i++)
        arb_div(c + i, a + i, b + i, prec);
}

void
_arb_vec_mid(arb_struct *y, const arb_struct *x, slong n)
{
    slong i;
    _arb_vec_set(y, x, n);
    for (i = 0; i < n; i++)
        mag_zero(arb_radref(y + i));
}

int
_arb_vec_intersection(arb_struct *c,
        const arb_struct *a, const arb_struct *b, slong n, slong prec)
{
    slong i;
    for (i = 0; i < n; i++)
        if (!arb_intersection(c + i, a + i, b + i, prec))
            return 0;
    return 1;
}

int
_arb_vec_contains(const arb_struct *a, const arb_struct *b, slong n)
{
    slong i;
    for (i = 0; i < n; i++)
        if (!arb_contains(a + i, b + i))
            return 0;
    return 1;
}

int
_arb_vec_contains_zero(const arb_struct *a, slong n)
{
    slong i;
    for (i = 0; i < n; i++)
        if (!arb_contains_zero(a + i));
            return 0;
    return 1;
}

int
_arb_vec_overlaps(const arb_struct *a, const arb_struct *b, slong n)
{
    slong i;
    for (i = 0; i < n; i++)
        if (!arb_overlaps(a + i, b + i))
            return 0;
    return 1;
}

int
_arb_vec_equal(const arb_struct *a, const arb_struct *b, slong n)
{
    slong i;
    for (i = 0; i < n; i++)
        if (!arb_equal(a + i, b + i))
            return 0;
    return 1;
}

int
_arb_vec_is_indeterminate(const arb_struct *v, slong n)
{
    slong i;
    for (i = 0; i < n; i++)
        if (arf_is_nan(arb_midref(v + i)))
            return 1;
    return 0;
}

/* implemented following _arb_vec_scalar_mul */
void
_arb_vec_scalar_sub(arb_ptr res, arb_srcptr vec,
        slong len, const arb_t c, slong prec)
{
    slong i;
    for (i = 0; i < len; i++)
        arb_sub(res + i, vec + i, c, prec);
}

/* from arb arb_poly zeta_series.c */
void
_arb_vec_printd(arb_srcptr vec, slong len, slong digits)
{
    slong i;
    for (i = 0; i < len; i++)
        arb_printd(vec + i, digits), flint_printf("\n");
}

void
_arb_vec_print(arb_srcptr vec, slong len)
{
    slong i;
    for (i = 0; i < len; i++)
        arb_print(vec + i), flint_printf("\n");
}
