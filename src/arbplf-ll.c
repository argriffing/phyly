/*
 * Use arbitrary precision matrix operations to compute a log likelihood.
 * The JSON format is used for both input and output.
 * Arbitrary precision is used only internally;
 * error bounds are not available for the input or the output,
 * but instead double precision floating point representation is used.
 * If the precision level used for intermediate calculations
 * is determined to be insufficient for reporting an "mpfr"-like
 * output (as indicated by the arb function arb_can_round_mpfr)
 * then an error is reported instead.
 *
 * The "probability_array" is a semantically flexible structure that defines
 * both the root prior distribution and the observations at the leaves.
 * Each site probability is the sum,
 * over all combinations of state assignments to nodes,
 * of the product of the state probabilities at nodes
 * times the product of transition probabilities on edges.
 *
 * For the log likelihood, the only available selection/aggregation
 * axis is the site axis. So the return value could consist of
 * an array of log likelihoods, or of summed, averaged, or linear
 * combinations of log likelihoods, optionally restricted to a
 * selection of sites.
 *
 * The output should be formatted in a way that is easily
 * readable as a data frame by the python module named pandas as follows:
 * >> import pandas as pd
 * >> f = open('output.json')
 * >> d = pd.read_json(f, orient='split', precise_float=True)
 *
 * input format:
 * {
 * "model_and_data" : {
 *  "edges" : [[a, b], [c, d], ...],                 (#edges, 2)
 *  "edge_rate_coefficients" : [a, b, ...],          (#edges, )
 *  "rate_matrix" : [[a, b, ...], [c, d, ...], ...], (#states, #states)
 *  "probability_array" : [...]                      (#sites, #nodes, #states)
 * },
 * "reductions" : [
 * {
 *  "columns" : ["site"],
 *  "selection" : [a, b, c, ...], (optional)
 *  "aggregation" : {"sum" | "avg" | [a, b, c, ...]} (optional)
 * }], (optional)
 * "working_precision" : a, (optional)
 * "sum_product_strategy" : {"brute_force" | "dynamic_programming"} (optional)
 * }
 *
 * output format (without aggregation of the "site" column):
 * {
 *  "columns" : ["site", "value"],
 *  "data" : [[a, b], [c, d], ..., [y, z]] (# selected sites)
 * }
 *
 * output format (with aggregation of the "site" column):
 * {
 *  "columns" : ["value"],
 *  "data" : [a]
 * }
 *
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

int main(void)
{
    /*
    json_hom_t hom;
    hom->userdata = NULL;
    hom->clear = NULL;
    hom->f = run;
    int result = run_json_script(hom);

    flint_cleanup();
    return result;
    */
    int i;
    i = 0;
    return add_two_ints(i, i);
}
