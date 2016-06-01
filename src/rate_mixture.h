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
    int n;
    double *rates;
    double *prior;
    enum rate_mixture_mode mode;
} rate_mixture_struct;
typedef rate_mixture_struct rate_mixture_t[1];

void rate_mixture_pre_init(rate_mixture_t x);
void rate_mixture_init(rate_mixture_t x, int n);
void rate_mixture_clear(rate_mixture_t x);
void rate_mixture_get_rate(arb_t rate, const rate_mixture_t x, slong idx);
void rate_mixture_get_prob(arb_t prob, const rate_mixture_t x, slong idx, slong prec);
slong rate_mixture_category_count(const rate_mixture_t x);
void rate_mixture_expectation(arb_t rate, const rate_mixture_t x, slong prec);


#ifdef __cplusplus
}
#endif

#endif
