#include "flint/flint.h"
#include "arb.h"
#include "arb_calc.h"
#include "acb_hypgeom.h"
#include "arb_vec_extras.h"

#include "gamma_heuristics.h"
#include "gamma_discretization.h"

#define GAMMA_DISC_DEBUG 0


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
        acb_hypgeom_gamma_lower(resc, sc, xc, regularized, prec);
        arb_set(res, acb_realref(resc));
        acb_clear(resc);
        acb_clear(sc);
        acb_clear(xc);
    }
}


/* provides the arb_calc_func_t interface */
static int
_arb_gamma_objective(arb_ptr out, const arb_t inp,
        void *param, slong order, slong prec)
{
    acb_t s;
    acb_ptr z, y;
    int regularized = 1;
    const quantile_args_struct *p = param;
    int zlen = FLINT_MIN(2, order);

    acb_init(s);
    z = _acb_vec_init(zlen);
    y = _acb_vec_init(order);

    arb_set(acb_realref(s), p->s);
    arb_set(acb_realref(z + 0), inp);
    if (zlen > 1)
        acb_one(z + 1);

    _acb_hypgeom_gamma_lower_series(y, s, z, zlen, regularized, order, prec);

    {
        slong i;
        for (i = 0; i < order; i++)
            arb_set(out + i, acb_realref(y + i));
    }

    arb_sub(out, out, p->c, prec);

    return 0;
}


static int
_sign(const arf_t q, const quantile_args_t p, slong prec)
{
    int sign;
    arb_t x, y, g;

    arb_init(x);
    arb_init(y);
    arb_init(g);

    arb_set_arf(x, q);
    /* for arb_calc callback compatibility we need to remove const */
    _arb_gamma_objective(y, x, (quantile_args_struct *) p, 1, prec);

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
_find_interval_bound(arf_interval_t r, const arf_t initial,
        int target_sign, const quantile_args_t p, slong prec)
{
    int result = ARB_CALC_SUCCESS;
    int sign;
    int logratio;
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
    arf_set(q, initial);
    logratio = target_sign;
    while (1)
    {
        sign = _sign(q, p, prec);
        if (GAMMA_DISC_DEBUG)
        {
            flint_printf("sign=%d target=%d prec=%wd\n",
                    sign, target_sign, prec);
        }
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
            arf_mul_2exp_si(q, q, logratio);
            logratio += target_sign;
        }
    }
    arf_clear(q);
    return result;
}


static int
isolate_positive_quantile(arf_interval_t r, const arf_t initial,
        const quantile_args_t p, slong prec)
{
    int result;

    arf_nan(&r->a);
    arf_nan(&r->b);

    /* find the lower bound of the interval */
    if (arf_is_nan(&r->a))
    {
        result = _find_interval_bound(r, initial, -1, p, prec);
        if (result != ARB_CALC_SUCCESS) return result;
    }

    /* find the upper bound of the interval */
    if (arf_is_nan(&r->b))
    {
        result = _find_interval_bound(r, initial, 1, p, prec);
        if (result != ARB_CALC_SUCCESS) return result;
    }

    return result;
}


void
gamma_quantile(arb_t res, slong k, slong n, const arb_t s, slong prec)
{
    int result;
    quantile_args_t p;
    arf_interval_t r, refined;
    slong maxiter;
    arf_t initial;
    arb_t guess;

    quantile_args_init(p);
    arf_interval_init(r);
    arf_interval_init(refined);
    arf_init(initial);
    arb_init(guess);

    arb_set(p->s, s);
    arb_set_si(p->c, k);
    arb_div_si(p->c, p->c, n, prec);

    /*
     * Compute r.
     * If r < (1 + s) / 5 then use (3.3) of
     * "Efficient and accurate algorithms for the computation
     * and inversion of the incomplete gamma function ratios"
     * to initialize a Newton-Raphson root search.
     * Otherwise use bisection.
     */
    result = gamma_quantile_temme_3_3(guess, p->s, p->c, prec);
    if (result == 0)
    {
        arf_set(initial, arb_midref(guess));
    }
    else
    {
        arf_one(initial);
    }

    result = isolate_positive_quantile(r, initial, p, prec);
    if (result != ARB_CALC_SUCCESS)
    {
        if (GAMMA_DISC_DEBUG)
        {
            flint_printf("(debug): failed to isolate quantile\n");
        }
        arb_indeterminate(res);
        goto finish;
    }
    else
    {
        if (GAMMA_DISC_DEBUG)
        {
            flint_printf("(debug): successfully isolated quantile\n");
        }
    }

    maxiter = 10000;
    result = arb_calc_refine_root_bisect(
            refined, _arb_gamma_objective, p, r, maxiter, prec);
    arf_interval_get_arb(res, refined, prec);

    if (GAMMA_DISC_DEBUG)
    {
        flint_printf("(debug): refined quantile:\n");
        arb_printd(res, 15); flint_printf("\n");
    }

finish:

    quantile_args_clear(p);
    arf_interval_clear(r);
    arf_interval_clear(refined);
    arf_clear(initial);
    arb_clear(guess);
}

void
gamma_expectile(arb_t res, const arb_t quantile, const arb_t s, slong prec)
{
    int regularized = 1;
    arb_t sp1;

    arb_init(sp1);
    arb_add_si(sp1, s, 1, prec);

    _arb_hypgeom_gamma_lower(res, sp1, quantile, regularized, prec);

    arb_clear(sp1);
}

void
gamma_rates(arb_struct *rates, slong n, const arb_t s, slong prec)
{
    int k;
    arb_struct *quantiles, *expectiles;

    quantiles = _arb_vec_init(n+1);
    expectiles = _arb_vec_init(n+1);
    arb_zero(quantiles + 0);
    arb_pos_inf(quantiles + n);

    for (k = 1; k < n; k++)
    {
        gamma_quantile(quantiles + k, k, n, s, prec);
    }

    if (GAMMA_DISC_DEBUG)
    {
        flint_printf("quantiles prec=%wd\n", prec);
        _arb_vec_printd(quantiles, n+1, 15);
    }
    for (k = 0; k < n+1; k++)
    {
        gamma_expectile(expectiles + k, quantiles + k, s, prec);
    }

    for (k = 0; k < n; k++)
    {
        arb_sub(rates + k, expectiles + k + 1, expectiles + k, prec);
        arb_mul_si(rates + k, rates + k, n, prec);
    }

    _arb_vec_clear(quantiles, n+1);
    _arb_vec_clear(expectiles, n+1);
}

void
normalized_median_gamma_rates(
        arb_struct *rates, slong n, const arb_t s, slong prec)
{
    int k;
    arb_struct *quantiles;

    quantiles = _arb_vec_init(n);
    for (k = 0; k < n; k++)
    {
        /*
         * If n == 4 then pick quantiles like o in
         * | o | o | o | o |
         * 0 1 2 3 4 5 6 7 8
         * Note that these are the medians of each quartile.
         */
        gamma_quantile(quantiles + k, 2*k + 1, 2*n, s, prec);
    }

    if (GAMMA_DISC_DEBUG)
    {
        flint_printf("quantiles prec=%wd\n", prec);
        _arb_vec_printd(quantiles, n, 15);
    }

    /*
     * Normalize so that the expected rate is 1.
     * This is equivalent to scaling the quantiles vector
     * so that its entries add up to n.
     */
    {
        _arb_vec_proportions(rates, quantiles, n, prec);
        _arb_vec_scalar_mul_si(rates, rates, n, n, prec);
    }

    _arb_vec_clear(quantiles, n);
}
