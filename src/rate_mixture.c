#include "arb.h"

#include "rate_mixture.h"
#include "gamma_discretization.h"
#include "arb_vec_extras.h"

void
custom_rate_mixture_pre_init(custom_rate_mixture_t x)
{
    x->n = 0;
    x->rates = NULL;
    x->prior = NULL;
    x->mode = RATE_MIXTURE_UNDEFINED;
}

void
custom_rate_mixture_init(custom_rate_mixture_t x, int n)
{
    x->n = n;
    x->rates = flint_malloc(n * sizeof(double));
    x->prior = flint_malloc(n * sizeof(double));
}

void
custom_rate_mixture_clear(custom_rate_mixture_t x)
{
    flint_free(x->rates);
    flint_free(x->prior);
    x->mode = RATE_MIXTURE_UNDEFINED;
}

void
custom_rate_mixture_get_rate(arb_t rate, const custom_rate_mixture_t x, slong idx)
{
    if (x->mode == RATE_MIXTURE_UNDEFINED)
    {
        flint_fprintf(stderr, "internal error: undefined rate mixture\n");
        abort();
    }
    else if (x->mode == RATE_MIXTURE_NONE)
    {
        arb_one(rate);
    }
    else if (x->mode == RATE_MIXTURE_UNIFORM || x->mode == RATE_MIXTURE_CUSTOM)
    {
        arb_set_d(rate, x->rates[idx]);
    }
    else
    {
        flint_fprintf(stderr, "internal error: "
                      "unrecognized rate mixture mode\n");
        abort();
    }
}

void
custom_rate_mixture_get_prob(arb_t prob, const custom_rate_mixture_t x,
        slong idx, slong prec)
{
    if (x->mode == RATE_MIXTURE_UNDEFINED)
    {
        flint_fprintf(stderr, "internal error: undefined rate mixture\n");
        abort();
    }
    else if (x->mode == RATE_MIXTURE_NONE)
    {
        arb_one(prob);
    }
    else if (x->mode == RATE_MIXTURE_UNIFORM)
    {
        /*
         * This code branch involves a division that could
         * unnecessarily lose exactness in some situations.
         */
        arb_set_si(prob, x->n);
        arb_inv(prob, prob, prec);
    }
    else if (x->mode == RATE_MIXTURE_CUSTOM)
    {
        arb_set_d(prob, x->prior[idx]);
    }
    else
    {
        flint_fprintf(stderr, "internal error: "
                      "unrecognized rate mixture mode\n");
        abort();
    }
}


void
custom_rate_mixture_expectation(arb_t rate, const custom_rate_mixture_t x, slong prec)
{
    if (x->mode == RATE_MIXTURE_UNDEFINED)
    {
        flint_fprintf(stderr, "internal error: undefined rate mixture\n");
        abort();
    }
    else if (x->mode == RATE_MIXTURE_NONE)
    {
        arb_one(rate);
    }
    else if (x->mode == RATE_MIXTURE_UNIFORM || x->mode == RATE_MIXTURE_CUSTOM)
    {
        slong i;
        arb_t tmp, tmpb;
        arb_init(tmp);
        arb_init(tmpb);
        arb_zero(rate);
        if (x->mode == RATE_MIXTURE_UNIFORM)
        {
            for (i = 0; i < x->n; i++)
            {
                arb_set_d(tmp, x->rates[i]);
                arb_add(rate, rate, tmp, prec);
            }
            arb_div_si(rate, rate, x->n, prec);
        }
        else if (x->mode == RATE_MIXTURE_CUSTOM)
        {
            for (i = 0; i < x->n; i++)
            {
                arb_set_d(tmp, x->rates[i]);
                arb_set_d(tmpb, x->prior[i]);
                arb_addmul(rate, tmp, tmpb, prec);
            }
        }
        arb_clear(tmp);
        arb_clear(tmpb);
    }
    else
    {
        flint_fprintf(stderr, "internal error: "
                      "unrecognized rate mixture mode\n");
        abort();
    }
}


/* gamma rate mixture functions */


void
gamma_rate_mixture_init(gamma_rate_mixture_t x)
{
    x->gamma_categories = 1;
    x->gamma_shape = 1;
    x->invariant_prior = 0;
}

void
gamma_rate_mixture_clear(gamma_rate_mixture_t x)
{
    gamma_rate_mixture_init(x);
}

slong
gamma_rate_mixture_category_count(const gamma_rate_mixture_t x)
{
    slong n = x->gamma_categories;
    if (x->invariant_prior)
        n++;
    return n;
}

void
gamma_rate_mixture_summarize(
        arb_ptr rate_mix_prior, arb_ptr rate_mix_rates, arb_ptr rate_mix_expect,
        const gamma_rate_mixture_t x, slong prec)
{
    slong i;
    arb_ptr r;

    arb_t p, q;
    arb_init(p);
    arb_init(q);
    arb_set_d(p, x->invariant_prior);
    arb_sub_si(q, p, 1, prec);
    arb_neg(q, q);

    /* the expectation is designed to be 1 */
    arb_one(rate_mix_expect);

    /* probabilities corresponding to gamma rate categories */
    for (i = 0; i < x->gamma_categories; i++)
    {
        r = rate_mix_prior + i;
        arb_div_si(r, q, x->gamma_categories, prec);
    }

    /* optional probability corresponding to the invariant category */
    if (x->invariant_prior)
    {
        r = rate_mix_prior + x->gamma_categories;
        arb_set(r, p);
    }

    /* set the discretized gamma rates */
    {
        arb_t s;
        arb_init(s);
        arb_set_d(s, x->gamma_shape);
        gamma_rates(rate_mix_rates, x->gamma_categories, s, prec);
        arb_clear(s);
    }

    /* normalize the rates to account for a possible invariant category */
    for (i = 0; i < x->gamma_categories; i++)
    {
        r = rate_mix_rates + i;
        arb_div(r, r, q, prec);
    }

    arb_clear(p);
    arb_clear(q);
}



/* generic rate mixture functions */


void
rate_mixture_pre_init(rate_mixture_t x)
{
    x->custom_mix = NULL;
    x->gamma_mix = NULL;
    x->mode = RATE_MIXTURE_UNDEFINED;
}

void
rate_mixture_clear(rate_mixture_t x)
{
    if (x->custom_mix)
    {
        custom_rate_mixture_clear(x->custom_mix);
    }
    if (x->gamma_mix)
    {
        gamma_rate_mixture_clear(x->gamma_mix);
    }
}

slong
rate_mixture_category_count(const rate_mixture_t x)
{
    if (x->mode == RATE_MIXTURE_UNDEFINED)
    {
        flint_fprintf(stderr, "internal error: undefined rate mixture\n");
        abort();
    }
    else if (x->mode == RATE_MIXTURE_NONE)
    {
        return 1;
    }
    else if (x->mode == RATE_MIXTURE_UNIFORM || x->mode == RATE_MIXTURE_CUSTOM)
    {
        return x->custom_mix->n;
    }
    else if (x->mode == RATE_MIXTURE_GAMMA)
    {
        return gamma_rate_mixture_category_count(x->gamma_mix);
    }
    else
    {
        flint_fprintf(stderr, "internal error: "
                      "unrecognized rate mixture mode\n");
        abort();
    }
}


void
rate_mixture_summarize(
        arb_ptr rate_mix_prior, arb_ptr rate_mix_rates, arb_ptr rate_mix_expect,
        const rate_mixture_t x, slong prec)
{
    if (x->mode == RATE_MIXTURE_NONE)
    {
        arb_one(rate_mix_prior);
        arb_one(rate_mix_rates);
        arb_one(rate_mix_expect);
    }
    else if (x->mode == RATE_MIXTURE_UNIFORM || x->mode == RATE_MIXTURE_CUSTOM)
    {
        slong i;
        slong n = x->custom_mix->n;
        custom_rate_mixture_expectation(rate_mix_expect, x->custom_mix, prec);
        for (i = 0; i < n; i++)
        {
            custom_rate_mixture_get_prob(
                    rate_mix_prior + i, x->custom_mix, i, prec);
            custom_rate_mixture_get_rate(
                    rate_mix_rates + i, x->custom_mix, i);
        }
    }
    else if (x->mode == RATE_MIXTURE_GAMMA)
    {
        gamma_rate_mixture_summarize(
                rate_mix_prior, rate_mix_rates, rate_mix_expect,
                x->gamma_mix, prec);

        /* debug */
        /*
        flint_printf("rate mix expect:\n");
        arb_printd(rate_mix_expect, 15);
        flint_printf("\n");

        flint_printf("rate mix prior:\n");
        _arb_vec_printd(rate_mix_prior,
                gamma_rate_mixture_category_count(x->gamma_mix), 15);
        flint_printf("\n");

        flint_printf("rate mix rates:\n");
        _arb_vec_printd(rate_mix_rates,
                gamma_rate_mixture_category_count(x->gamma_mix), 15);
        flint_printf("\n");
        */
    }
    else
    {
        abort(); /* assert */
    }
}
