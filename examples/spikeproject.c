/*
 * Use arbitrary precision matrix operations to compute a likelihood.
 * This uses a manually executed Felsenstein pruning algorithm
 * on a hardcoded tree.
 */

#include "flint.h"
#include "arb_mat.h"

void
_arb_mat_mul_entrywise(arb_mat_t c, arb_mat_t a, arb_mat_t b, slong prec)
{
    slong i, j, nr, nc;

    nr = arb_mat_nrows(a);
    nc = arb_mat_ncols(a);

    for (i = 0; i < nr; i++)
    {
        for (j = 0; j < nc; j++)
        {
            arb_mul(arb_mat_entry(c, i, j),
                    arb_mat_entry(a, i, j),
                    arb_mat_entry(b, i, j), prec);
        }
    }
}

void
_jukes_cantor_rate_matrix(fmpq_mat_t Q)
{
    slong i, j;
    fmpq *p;
    for (i = 0; i < 4; i++)
    {
        for (j = 0; j < 4; j++)
        {
            p = fmpq_mat_entry(Q, i, j);
            if (i == j)
            {
                fmpq_one(p);
                fmpq_neg(p, p);
            }
            else
            {
                fmpq_set_si(p, 1, 3);
            }
        }
    }
}

void
_fmpq_mat_scalar_mul_fmpq(fmpq_mat_t dest, fmpq_mat_t src, fmpq_t x)
{
    slong nr, nc, i, j;
    nr = fmpq_mat_nrows(src);
    nc = fmpq_mat_ncols(src);
    for (i = 0; i < nr; i++)
    {
        for (j = 0; j < nc; j++)
        {
            fmpq_mul(fmpq_mat_entry(dest, i, j),
                     fmpq_mat_entry(src, i, j), x);
        }
    }
}

void
_prune_helper(arb_mat_t res, fmpq_mat_t Q, slong p, ulong q,
              arb_mat_t mat, slong prec)
{
    arb_mat_t R, P;
    fmpq_mat_t Qt;
    fmpq_t t;

    /* initialize the exact branch length */
    fmpq_init(t);
    fmpq_set_si(t, p, q);

    /*
    flint_printf("branch length: ");
    fmpq_print(t);
    flint_printf("\n");
    */

    /* initialize the exact rate matrix */
    fmpq_mat_init(Qt, 4, 4);
    _fmpq_mat_scalar_mul_fmpq(Qt, Q, t);

    /*
    flint_printf("scaled rational rate matrix:\n");
    fmpq_mat_print(Qt);
    */

    /* initialize the arbitrary precision rate matrix */
    arb_mat_init(R, 4, 4);
    arb_mat_set_fmpq_mat(R, Qt, prec);

    /* initialize the matrix exponential */
    arb_mat_init(P, 4, 4);
    arb_mat_exp(P, R, prec);

    /*
    flint_printf("probability transition matrix:\n");
    arb_mat_printd(P, 15);
    */

    /* compute the matrix exponential matrix product */
    arb_mat_mul(res, P, mat, prec);

    arb_mat_clear(R);
    arb_mat_clear(P);
    fmpq_mat_clear(Qt);
    fmpq_clear(t);
}

void
_prune(arb_mat_t Lc, fmpq_mat_t Q,
       arb_mat_t La, slong pa, ulong qa,
       arb_mat_t Lb, slong pb, ulong qb, slong prec)
{
    _prune_helper(La, Q, pa, qa, La, prec);
    _prune_helper(Lb, Q, pb, qb, Lb, prec);
    _arb_mat_mul_entrywise(Lc, La, Lb, prec);
}

int
main(int argc, char *argv[])
{
    slong prec, i, j;
    /*
    slong digits, 
    */
    arb_mat_struct leaf[5];
    fmpq_mat_t Q;
    arb_mat_t prior;
    arb_mat_t final;

    /*
    digits = 100;
    prec = digits * 3.3219280948873623 + 5;
    */
    prec = 8192;

    flint_printf("working precision: %ld\n", prec);

    flint_printf("initializing leaf states...\n");
    for (i = 0; i < 5; i++)
    {
        arb_mat_init(leaf + i, 4, 1);
        arb_mat_zero(leaf + i);
    }
    arb_one(arb_mat_entry(leaf+0, 0, 0)); /* A */
    arb_one(arb_mat_entry(leaf+1, 1, 0)); /* C */
    arb_one(arb_mat_entry(leaf+2, 1, 0)); /* C */
    arb_one(arb_mat_entry(leaf+3, 1, 0)); /* C */
    arb_one(arb_mat_entry(leaf+4, 2, 0)); /* G */


    /* initialize the prior row vector [[1/4, 1/4, 1/4, 1/4]] */
    flint_printf("initializing prior...\n");
    arb_mat_init(prior, 1, 4);
    for (i = 0; i < 4; i++)
    {
        arb_one(arb_mat_entry(prior, 0, i));
        arb_mul_2exp_si(arb_mat_entry(prior, 0, i),
                        arb_mat_entry(prior, 0, i), -2);
    }

    /* initialize the 4x4 Jukes-Cantor rate matrix */
    flint_printf("initializing rate matrix...\n");
    fmpq_mat_init(Q, 4, 4);
    _jukes_cantor_rate_matrix(Q);

    /*
    flint_printf("rational rate matrix:\n");
    fmpq_mat_print(Q);
    */

    /* prune the tree in a hardcoded order, with hardcoded branch lengths */
    flint_printf("pruning...\n");
    _prune(leaf+0, Q, leaf+0, 1, 100, leaf+1, 1, 5, prec);
    _prune(leaf+3, Q, leaf+3, 3, 10, leaf+4, 1, 50, prec);
    _prune(leaf+2, Q, leaf+2, 3, 10, leaf+3, 1, 20, prec);
    _prune(leaf+0, Q, leaf+0, 1, 20, leaf+2, 2, 20, prec);

    /* compute a final dot product involving the prior distribution */
    flint_printf("taking final dot product...\n");
    arb_mat_init(final, 1, 1);
    arb_mat_mul(final, prior, leaf+0, prec);

    /* report the likelihood */
    flint_printf("likelihood: ");
    arb_printn(arb_mat_entry(final, 0, 0), 50000, ARB_STR_CONDENSE * 20);
    flint_printf("\n");
    /*
    flint_printf("likelihood: ");
    arb_print(arb_mat_entry(final, 0, 0));
    flint_printf("\n");
    */

    /* clear */
    arb_mat_clear(final);
    arb_mat_clear(prior);
    fmpq_mat_clear(Q);
    for (i = 0; i < 5; i++)
    {
        arb_mat_clear(leaf + i);
    }

    return 0;
}
