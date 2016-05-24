#include "util.h"
#include "arb_mat_extras.h"
#include "distribution.h"

void
distribution_init(distribution_t d, slong len)
{
    d->n = len;
    d->p = _arb_vec_init(len);
    d->simplex_constraint = 0;
}

void
distribution_set_col(distribution_t d, const arb_mat_t A, slong col,
        int simplex_constraint)
{
    slong i;
    if (arb_mat_nrows(A) != d->n) abort(); /* assert */
    d->simplex_constraint = simplex_constraint;
    for (i = 0; i < d->n; i++)
        arb_set(d->p + i, arb_mat_entry(A, i, col));
}

void
distribution_set_vec(distribution_t d, const arb_struct *vec, slong len,
        int simplex_constraint)
{
    if (len != d->n) abort(); /* assert */
    d->simplex_constraint = simplex_constraint;
    _arb_vec_set(d->p, vec, len);
}

int
distribution_is_finite(const distribution_t d)
{
    return _arb_vec_is_finite(d->p, d->n);
}

void
distribution_clear(distribution_t d)
{
    _arb_vec_clear(d->p, d->n);
}

arb_ptr
distribution_entry(distribution_t d, slong i)
{
    return d->p + i;
}

arb_srcptr
distribution_srcentry(const distribution_t d, slong i)
{
    return d->p + i;
}

void
distribution_expectation_of_col(arb_t out, const distribution_struct *d,
        const arb_mat_t g, slong col, slong prec)
{
    slong i;
    /*
     * If the distribution is NULL
     * then return the sum of conditional expectations.
     */
    if (d == NULL)
    {
        arb_zero(out);
        for (i = 0; i < arb_mat_nrows(g); i++)
        {
            arb_add(out, out, arb_mat_entry(g, i, col), prec);
        }
    }
    else
    {
        /*
         * todo: this could be improved by requiring only
         * that the column entries be constant across
         * the nonzero support of the distribution.
         */
        if (d->simplex_constraint &&
            _arb_mat_col_is_constant(g, col) &&
            distribution_is_finite(d))
        {
            arb_set(out, arb_mat_entry(g, 0, col));
        }
        else
        {
            if (arb_mat_nrows(g) != d->n) abort(); /* assert */
            arb_zero(out);
            for (i = 0; i < d->n; i++)
            {
                arb_addmul(out,
                        distribution_srcentry(d, i),
                        arb_mat_entry(g, i, col), prec);
            }
        }
    }
}
