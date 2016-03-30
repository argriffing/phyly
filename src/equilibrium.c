#include "util.h"
#include "equilibrium.h"

void _arb_vec_rate_matrix_equilibrium(
        arb_struct *p, const arb_mat_t Q, slong prec)
{
    slong i, n;
    arb_mat_t x;
    n = arb_mat_nrows(Q);
    arb_mat_init(x, n, 1);
    _arb_mat_rate_matrix_equilibrium(x, Q, prec);
    for (i = 0; i < n; i++)
        arb_set(p + i, arb_mat_entry(x, i, 0));
    arb_mat_clear(x);
}

void _arb_mat_rate_matrix_equilibrium(
        arb_mat_t p, const arb_mat_t Q, slong prec)
{
    slong i, j, n;
    arb_mat_t R;
    arb_mat_t x, b;
    arb_struct *exit_rates;
    n = arb_mat_nrows(Q);

    /* initialize exit rates */
    exit_rates = _arb_vec_init(n);
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (i != j)
            {
                arb_add(exit_rates + i, exit_rates + i,
                        arb_mat_entry(Q, i, j), prec);
            }
        }
    }

    /* initialize the right-hand-side */
    arb_mat_init(b, n+1, 1);
    _arb_mat_ones(b);

    /* initialize the R matrix as [Q^T e; e^t 0] */
    arb_mat_init(R, n+1, n+1);
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (i != j)
            {
                arb_set(arb_mat_entry(R, i, j), arb_mat_entry(Q, j, i));
            }
        }
    }
    for (i = 0; i < n; i++)
    {
        arb_set(arb_mat_entry(R, i, i), exit_rates + i);
        arb_one(arb_mat_entry(R, n, i));
        arb_one(arb_mat_entry(R, i, n));
    }

    /* solve the system */
    arb_mat_init(x, n+1, 1);
    arb_mat_solve(x, R, b, prec);

    /* copy the solution to the output matrix */
    for (i = 0; i < n; i++)
    {
        arb_set(arb_mat_entry(p, i, 0), arb_mat_entry(x, i, 0));
    }

    _arb_vec_clear(exit_rates, n);
    arb_mat_clear(R);
    arb_mat_clear(x);
    arb_mat_clear(b);
}
