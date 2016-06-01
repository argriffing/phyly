#include "arb.h"

#include "rate_mixture.h"

void
rate_mixture_pre_init(rate_mixture_t x)
{
    x->n = 0;
    x->rates = NULL;
    x->prior = NULL;
    x->mode = RATE_MIXTURE_UNDEFINED;
}

void
rate_mixture_init(rate_mixture_t x, int n)
{
    x->n = n;
    x->rates = flint_malloc(n * sizeof(double));
    x->prior = flint_malloc(n * sizeof(double));
}

void
rate_mixture_clear(rate_mixture_t x)
{
    flint_free(x->rates);
    flint_free(x->prior);
    x->mode = RATE_MIXTURE_UNDEFINED;
}

void
rate_mixture_get_rate(arb_t rate, const rate_mixture_t x, slong idx)
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
rate_mixture_get_prob(arb_t prob, const rate_mixture_t x,
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
        return x->n;
    }
    else
    {
        flint_fprintf(stderr, "internal error: "
                      "unrecognized rate mixture mode\n");
        abort();
    }
}

void
rate_mixture_expectation(arb_t rate, const rate_mixture_t x, slong prec)
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
