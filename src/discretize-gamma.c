#include "flint/flint.h"
#include "arb.h"
#include "arb_calc.h"

/*
 * TODO: These parameters should be provided on the commmand line
 *       instead of being hardcoded.
 *
 * According to R, one of the intermediate values should be:
 * > qgamma(0.25, shape=shape)
 * [1] 0.00048270865921202978522
 *
 */
/* #define SHAPE 0.19242344607262146 */
#define SHAPE 0.5
#define NCATS 4


static void
_arb_hypgeom_gamma_lower(arb_t res, const arb_t s, const arb_t x,
        int regularized, slong prec)
{
    if (regularized == 1 && arf_is_pos_inf(arb_midref(x)))
    {
        arb_one(res);
    }
    else
    {
        acb_t resc, sc, xc;
        acb_init(resc);
        acb_init(sc);
        acb_init(xc);
        arb_set(acb_realref(sc), s);
        arb_set(acb_realref(xc), x);

        //flint_printf("acb_hypgeom_gamma_lower input:\n");
        //acb_printd(xc, 15); flint_printf("\n");

        acb_hypgeom_gamma_lower(resc, sc, xc, regularized, prec);

        //flint_printf("acb_hypgeom_gamma_lower output:\n");
        //acb_printd(resc, 15); flint_printf("\n");

        arb_set(res, acb_realref(resc));

        acb_clear(resc);
        acb_clear(sc);
        acb_clear(xc);
    }
}

typedef struct
{
    arb_t s; /* gamma shape parameter */
    arb_t c; /* cdf value at quantile of interest */
} quantile_args_struct;
typedef quantile_args_struct quantile_args_t[1];

static void
quantile_args_init(quantile_args_t p)
{
    arb_init(p->s);
    arb_init(p->c);
}

static void
quantile_args_clear(quantile_args_t p)
{
    arb_clear(p->s);
    arb_clear(p->c);
}


/* provides the arb_calc_func_t interface */
static int
_arb_gamma_objective(arb_t out, const arb_t inp,
        void *param, slong order, slong prec)
{
    int regularized = 1;
    const quantile_args_struct *p = param;

    // flint_printf("x in:\n");
    // arb_printd(inp, 15); flint_printf("\n");

    // flint_printf("s in:\n");
    // arb_printd(p->s, 15); flint_printf("\n");

    _arb_hypgeom_gamma_lower(out, p->s, inp, regularized, prec);
    // flint_printf("out before subtraction:\n");
    // arb_printd(out, 15); flint_printf("\n");

    arb_sub(out, out, p->c, prec);
    // flint_printf("out after subtraction:\n");
    // arb_printd(out, 15); flint_printf("\n");

    // flint_printf("\n");

    return 0;
}

int
_sign(const arf_t q, const quantile_args_t p, slong prec)
{
    int sign;
    int order = 1;
    int regularized = 1;
    int result;
    arb_t x, y, g;

    arb_init(x);
    arb_init(y);
    arb_init(g);

    arb_set_arf(x, q);
    /* for arb_calc callback compatibility we need to remove const */
    result = _arb_gamma_objective(y, x, (quantile_args_struct *) p, 1, prec);

    arb_sgn(g, y);
    if (arb_contains_zero(g))
    {
        sign = 0;
    }
    else
    {
        sign = arf_sgn(arb_midref(g));
    }

    arb_clear(x);
    arb_clear(y);
    arb_clear(g);

    return sign;
}

/* return an ARB_CALC_ code */
static int
_find_interval_bound(arf_interval_t r, int target_sign,
        const quantile_args_t p, slong prec)
{
    int result = ARB_CALC_SUCCESS;
    int sign;
    arf_t q;
    arf_struct *head, *tail;
    if (target_sign == 1)
    {
        head = &(r->b);
        tail = &(r->a);
    }
    else
    {
        head = &(r->a);
        tail = &(r->b);
    }
    arf_init(q);
    arf_one(q);
    while (1)
    {
        sign = _sign(q, p, prec);
        //flint_printf("sign=%d target=%d prec=%wd\n", sign, target_sign, prec);
        if (sign == 0)
        {
            /* the sign is ambiguous */
            result = ARB_CALC_IMPRECISE_INPUT;
            break;
        }
        else if (sign == target_sign)
        {
            /* found the target sign */
            result = ARB_CALC_SUCCESS;
            arf_set(head, q);
            break;
        }
        else
        {
            /* keep searching, towards zero or towards infinity */
            arf_set(tail, q);
            arf_mul_2exp_si(q, q, target_sign);
        }
    }
    arf_clear(q);
    return result;
}


int
isolate_positive_quantile(arf_interval_t r,
        const quantile_args_t p, slong prec)
{
    int result;

    arf_nan(&r->a);
    arf_nan(&r->b);

    /* find the lower bound of the interval */
    if (arf_is_nan(&r->a))
    {
        result = _find_interval_bound(r, -1, p, prec);
        if (result != ARB_CALC_SUCCESS) return result;
    }

    /* find the upper bound of the interval */
    if (arf_is_nan(&r->b))
    {
        result = _find_interval_bound(r, 1, p, prec);
        if (result != ARB_CALC_SUCCESS) return result;
    }

    return result;
}

int
compute_quantile(arb_t res, int k, slong prec)
{
    int result;
    quantile_args_t p;
    arf_interval_t r, refined;
    slong maxiter;

    quantile_args_init(p);
    arf_interval_init(r);
    arf_interval_init(refined);

    arb_set_d(p->s, SHAPE);
    arb_set_si(p->c, k);
    arb_div_si(p->c, p->c, NCATS, prec);

    result = isolate_positive_quantile(r, p, prec);
    if (result != ARB_CALC_SUCCESS)
    {
        arb_indeterminate(res);
        goto finish;
    }

    maxiter = 1000;
    result = arb_calc_refine_root_bisect(
            refined, _arb_gamma_objective, p, r, maxiter, prec);
    arf_interval_get_arb(res, refined, prec);

finish:

    quantile_args_clear(p);
    arf_interval_clear(r);
    arf_interval_clear(refined);

    return result;
}

int
compute_expectile(arb_t expectile, const arb_t quantile, slong prec)
{
    int result;
    int regularized = 1;
    arb_t sp1;

    arb_init(sp1);
    arb_set_d(sp1, SHAPE);
    arb_add_si(sp1, sp1, 1, prec);

    _arb_hypgeom_gamma_lower(expectile, sp1, quantile, regularized, prec);

    arb_clear(sp1);

    return result;
}

int
main()
{
    int k;
    arb_struct *quantiles, *expectiles, *rates;
    slong prec = 4;

    quantiles = _arb_vec_init(NCATS+1);
    expectiles = _arb_vec_init(NCATS+1);
    rates = _arb_vec_init(NCATS);
    arb_zero(quantiles + 0);
    arb_pos_inf(quantiles + NCATS);

    while (1)
    {
        slong worstprec;
        for (k = 1; k < NCATS; k++)
        {
            compute_quantile(quantiles + k, k, prec);
        }
        for (k = 0; k < NCATS+1; k++)
        {
            compute_expectile(expectiles + k, quantiles + k, prec);
            //arb_printd(expectiles + k, 15); flint_printf("\n");
        }
        //flint_printf("\n");

        for (k = 0; k < NCATS; k++)
        {
            slong p;
            arb_sub(rates + k, expectiles + k + 1, expectiles + k, prec);
            arb_mul_si(rates + k, rates + k, NCATS, prec);
            p = arb_rel_accuracy_bits(rates + k);
            if ((k == 0) || p < worstprec)
                worstprec = p;
        }
        if (worstprec > 53)
            break;
        prec <<= 1;
    }

    flint_printf("quantiles:\n");
    for (k = 0; k < NCATS+1; k++)
    {
        arb_printd(quantiles + k, 15);
        flint_printf("\n");
    }

    flint_printf("expectiles:\n");
    for (k = 0; k < NCATS+1; k++)
    {
        arb_printd(expectiles + k, 15);
        flint_printf("\n");
    }

    flint_printf("rates:\n");
    for (k = 0; k < NCATS; k++)
    {
        arb_printd(rates + k, 15);
        flint_printf("\n");
    }

    _arb_vec_clear(quantiles, NCATS+1);
    _arb_vec_clear(expectiles, NCATS+1);
    _arb_vec_clear(rates, NCATS);

    return 0;
}
