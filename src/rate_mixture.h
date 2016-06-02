#ifndef RATE_MIXTURE_H
#define RATE_MIXTURE_H

#include "arb.h"

#ifdef __cplusplus
extern "C" {
#endif


enum rate_mixture_mode {
    RATE_MIXTURE_UNDEFINED,
    RATE_MIXTURE_NONE,
    RATE_MIXTURE_CUSTOM,
    RATE_MIXTURE_UNIFORM,
    RATE_MIXTURE_GAMMA};

typedef struct
{
    int gamma_categories;
    double gamma_shape;
    double invariant_prior;
} gamma_rate_mixture_struct;
typedef gamma_rate_mixture_struct gamma_rate_mixture_t[1];

void gamma_rate_mixture_init(gamma_rate_mixture_t x, int n);
void gamma_rate_mixture_clear(gamma_rate_mixture_t x);
slong gamma_rate_mixture_category_count(const gamma_rate_mixture_t x);
void gamma_rate_mixture_summarize(
        arb_ptr rate_mix_prior, arb_ptr rate_mix_rates, arb_ptr rate_mix_expect,
        const gamma_rate_mixture_t x, slong prec);


typedef struct
{
    int n;
    double *rates;
    double *prior;
    enum rate_mixture_mode mode;
} custom_rate_mixture_struct;
typedef custom_rate_mixture_struct custom_rate_mixture_t[1];

void custom_rate_mixture_pre_init(custom_rate_mixture_t x);
void custom_rate_mixture_init(custom_rate_mixture_t x, int n);
void custom_rate_mixture_clear(custom_rate_mixture_t x);
void custom_rate_mixture_get_prob(
        arb_t prob, const custom_rate_mixture_t x, slong idx, slong prec);
void custom_rate_mixture_get_rate(
        arb_t rate, const custom_rate_mixture_t x, slong idx);
void custom_rate_mixture_expectation(
        arb_t rate, const custom_rate_mixture_t x, slong prec);


typedef struct
{
    gamma_rate_mixture_struct * gamma_mix;
    custom_rate_mixture_struct * custom_mix;
    enum rate_mixture_mode mode;
} rate_mixture_struct;
typedef rate_mixture_struct rate_mixture_t[1];

void rate_mixture_pre_init(rate_mixture_t x);
void rate_mixture_clear(rate_mixture_t x);
slong rate_mixture_category_count(const rate_mixture_t x);
void rate_mixture_summarize(
        arb_ptr rate_mix_prior, arb_ptr rate_mix_rates, arb_ptr rate_mix_expect,
        const rate_mixture_t x, slong prec);


#ifdef __cplusplus
}
#endif

#endif
