#include "flint/flint.h"

#include "arb.h"
#include "arb_mat.h"

#include "util.h"
#include "arb_mat_extras.h"

arb_mat_struct *
_arb_mat_vec_init(slong nrows, slong ncols, slong len)
{
    slong i;
    arb_mat_struct *vec = flint_malloc(len * sizeof(arb_mat_struct));
    for (i = 0; i < len; i++)
    {
        arb_mat_init(vec + i, nrows, ncols);
    }
    return vec;
}

void
_arb_mat_vec_clear(arb_mat_struct *vec, slong len)
{
    slong i;
    if (vec)
    {
        for (i = 0; i < len; i++)
        {
            arb_mat_clear(vec + i);
        }
        flint_free(vec);
    }
}

int
_arb_mat_col_is_constant(const arb_mat_t A, slong j)
{
    slong i;
    arb_srcptr ref;
    if (arb_mat_is_empty(A))
        return 1;
    ref = arb_mat_entry(A, 0, j);
    if (!arb_is_exact(ref))
        return 0;
    for (i = 0; i < arb_mat_nrows(A); i++)
    {
        if (!arb_equal(arb_mat_entry(A, i, j), ref))
            return 0;
    }
    return 1;
}

/* Assume that each row of A sums to 1 */
void
_arb_mat_mul_stochastic(
        arb_mat_t C, const arb_mat_t A, const arb_mat_t B, slong prec)
{
    slong ar, ac, br, bc, i, j, k;

    ar = arb_mat_nrows(A);
    ac = arb_mat_ncols(A);
    br = arb_mat_nrows(B);
    bc = arb_mat_ncols(B);

    if (ac != br || ar != arb_mat_nrows(C) || bc != arb_mat_ncols(C))
    {
        flint_printf("arb_mat_mul: incompatible dimensions\n");
        abort();
    }

    if (br == 0)
    {
        arb_mat_zero(C);
        return;
    }

    if (A == C || B == C)
    {
        arb_mat_t T;
        arb_mat_init(T, ar, bc);
        _arb_mat_mul_stochastic(T, A, B, prec);
        arb_mat_swap(T, C);
        arb_mat_clear(T);
        return;
    }

    for (j = 0; j < bc; j++)
    {
        if (_arb_mat_col_is_constant(B, j))
        {
            for (i = 0; i < ar; i++)
            {
                arb_set(arb_mat_entry(C, i, j), arb_mat_entry(B, 0, j));
            }
        }
        else
        {
            for (i = 0; i < ar; i++)
            {
                arb_mul(arb_mat_entry(C, i, j),
                          arb_mat_entry(A, i, 0),
                          arb_mat_entry(B, 0, j), prec);

                for (k = 1; k < br; k++)
                {
                    arb_addmul(arb_mat_entry(C, i, j),
                                 arb_mat_entry(A, i, k),
                                 arb_mat_entry(B, k, j), prec);
                }
            }
        }
    }
}

/* Assume that each row of A sums to 0 */
void
_arb_mat_mul_rate_matrix(
        arb_mat_t C, const arb_mat_t A, const arb_mat_t B, slong prec)
{
    slong ar, ac, br, bc, i, j, k;

    ar = arb_mat_nrows(A);
    ac = arb_mat_ncols(A);
    br = arb_mat_nrows(B);
    bc = arb_mat_ncols(B);

    if (ac != br || ar != arb_mat_nrows(C) || bc != arb_mat_ncols(C))
    {
        flint_printf("arb_mat_mul: incompatible dimensions\n");
        abort();
    }

    if (br == 0)
    {
        arb_mat_zero(C);
        return;
    }

    if (A == C || B == C)
    {
        arb_mat_t T;
        arb_mat_init(T, ar, bc);
        _arb_mat_mul_rate_matrix(T, A, B, prec);
        arb_mat_swap(T, C);
        arb_mat_clear(T);
        return;
    }

    for (j = 0; j < bc; j++)
    {
        if (_arb_mat_col_is_constant(B, j))
        {
            for (i = 0; i < ar; i++)
            {
                arb_zero(arb_mat_entry(C, i, j));
            }
        }
        else
        {
            for (i = 0; i < ar; i++)
            {
                arb_mul(arb_mat_entry(C, i, j),
                          arb_mat_entry(A, i, 0),
                          arb_mat_entry(B, 0, j), prec);

                for (k = 1; k < br; k++)
                {
                    arb_addmul(arb_mat_entry(C, i, j),
                                 arb_mat_entry(A, i, k),
                                 arb_mat_entry(B, k, j), prec);
                }
            }
        }
    }
}

int
_arb_mat_is_finite(const arb_mat_t A)
{
    slong i, j;
    for (i = 0; i < arb_mat_nrows(A); i++)
        for (j = 0; j < arb_mat_ncols(A); j++)
            if (!arb_is_finite(arb_mat_entry(A, i, j)))
                return 0;
    return 1;
}
