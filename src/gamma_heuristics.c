#include "flint/flint.h"
#include "arb.h"
#include "arb_vec_extras.h"

#include "gamma_heuristics.h"

#define GAMMA_HEUR_DEBUG 0

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
int
gamma_quantile_temme_3_3(arb_t res, const arb_t s, const arb_t p, slong prec)
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

        if (GAMMA_HEUR_DEBUG)
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
