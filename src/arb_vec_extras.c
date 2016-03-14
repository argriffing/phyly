#include "flint/flint.h"

#include "arb.h"

#include "arb_vec_extras.h"

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
_arb_vec_overlaps(const arb_struct *a, const arb_struct *b, slong n)
{
    slong i;
    for (i = 0; i < n; i++)
        if (!arb_overlaps(a + i, b + i))
            return 0;
    return 1;
}

/* from arb arb_poly zeta_series.c */
void
_arb_vec_printd(arb_srcptr vec, slong len, slong digits)
{
    slong i;
    for (i = 0; i < len; i++)
        arb_printd(vec + i, digits), flint_printf("\n");
}
