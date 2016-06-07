#include "flint/flint.h"
#include "arb.h"
#include "arb_calc.h"
#include "acb_hypgeom.h"
#include "arb_vec_extras.h"

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


/* fixme: deprecated */
/* provides the arb_calc_func_t interface */
static int
old_arb_gamma_objective(arb_t out, const arb_t inp,
        void *param, slong order, slong prec)
{
    int regularized = 1;
    const quantile_args_struct *p = param;
    _arb_hypgeom_gamma_lower(out, p->s, inp, regularized, prec);
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

/* a helper function for gamma quantiles */
static void
_temme_r(arb_t r, const arb_t a, const arb_t p, slong prec)
{
    /* (p*gamma(1+a))^(1/a) = exp((log(p) + lgamma(1+a))/a) */
    arb_t logp;
    arb_init(logp);

    if (r == p || r == a) abort(); /* assert */

    arb_log(logp, p, prec);
    arb_add_si(r, a, 1, prec);
    arb_lgamma(r, r, prec);
    arb_add(r, r, logp, prec);
    arb_div(r, r, a, prec);
    arb_exp(r, r, prec);

    arb_clear(logp);
}

static void
_si_poly(arb_t res, const slong *coeffs, slong len, const arb_t x, slong prec)
{
    slong i;
    arb_zero(res);
    for (i = len-1; i > -1; i--)
    {
        arb_mul(res, res, x, prec);
        arb_add_si(res, res, coeffs[i], prec);
    }
}

static void
_temme_3_3(arb_t res, const arb_t a_in, const arb_t r, slong prec)
{
    slong i;
    arb_ptr u;
    arb_struct *c = _arb_vec_init(6);
    arb_struct *a = _arb_vec_init(5);
    for (i = 0; i < 5; i++)
        arb_add_si(a + i, a_in, i, prec);

    /* c0 */
    arb_zero(c + 0);

    /* c1 */
    arb_one(c + 1);

    /* c2 */
    u = c + 2;
    arb_one(u);
    arb_div(u, u, a + 1, prec);

    /* c3 */
    u = c + 3;
    {
        slong coeffs[] = {5, 3};
        _si_poly(u, coeffs, 2, a, prec);
    }
    arb_div_si(u, u, 2, prec);
    arb_div(u, u, a + 1, prec);
    arb_div(u, u, a + 1, prec);
    arb_div(u, u, a + 2, prec);

    /* c4 */
    u = c + 4;
    {
        slong coeffs[] = {31, 33, 8};
        _si_poly(u, coeffs, 3, a, prec);
    }
    arb_div_si(u, u, 3, prec);
    arb_div(u, u, a + 1, prec);
    arb_div(u, u, a + 1, prec);
    arb_div(u, u, a + 1, prec);
    arb_div(u, u, a + 2, prec);
    arb_div(u, u, a + 3, prec);

    /* c5 */
    u = c + 5;
    {
        slong coeffs[] = {2888, 5661, 3971, 1179, 125};
        _si_poly(u, coeffs, 5, a, prec);
    }
    arb_div_si(u, u, 24, prec);
    arb_div(u, u, a + 1, prec);
    arb_div(u, u, a + 1, prec);
    arb_div(u, u, a + 1, prec);
    arb_div(u, u, a + 1, prec);
    arb_div(u, u, a + 2, prec);
    arb_div(u, u, a + 2, prec);
    arb_div(u, u, a + 3, prec);
    arb_div(u, u, a + 4, prec);

    {
        slong i;
        arb_zero(res);
        for (i = 6-1; i > -1; i--)
        {
            arb_mul(res, res, r, prec);
            arb_add(res, res, c + i, prec);
        }
    }

    _arb_vec_clear(a, 5);
    _arb_vec_clear(c, 6);
}

/* return 0 if there is no error */
static int
_gamma_quantile_temme_3_3(arb_t res, const arb_t s, const arb_t p, slong prec)
{
    int result;
    arb_t r, threshold;
    arb_init(r);
    arb_init(threshold);
    _temme_r(r, s, p, prec);
    arb_add_si(threshold, s, 1, prec);
    arb_div_si(threshold, threshold, 5, prec);
    if (arb_lt(r, threshold))
    {
        _temme_3_3(res, s, r, prec);
        /*
         * Multiply by an irrational number
         * so that we can determine which side of the root we are on.
         */
        {
            arb_t sqrt2;
            arb_init(sqrt2);
            arb_sqrt_ui(sqrt2, 2, prec);
            arb_mul(res, res, sqrt2, prec);
            arb_clear(sqrt2);
        }

        if (GAMMA_DISC_DEBUG)
        {
            flint_printf("debug (Eq. 3.3):\n");
            flint_printf("prec=%wd\n", prec);
            flint_printf("s:\n");
            arb_printd(s, 15); flint_printf("\n");
            flint_printf("p:\n");
            arb_printd(p, 15); flint_printf("\n");
            flint_printf("initial (temme Eq. 3.3 times sqrt 2):\n");
            arb_printd(res, 15); flint_printf("\n");
        }

        result = 0;
    }
    else
    {
        result = -1;
    }
    arb_clear(r);
    arb_clear(threshold);
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
    result = _gamma_quantile_temme_3_3(guess, p->s, p->c, prec);
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
