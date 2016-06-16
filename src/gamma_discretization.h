#ifndef GAMMA_DISCRETIZATION_H
#define GAMMA_DISCRETIZATION_H

#include "flint/flint.h"
#include "arb.h"
#include "arb_calc.h"


#ifdef __cplusplus
extern "C" {
#endif

void gamma_rates(arb_struct *rates, slong n, const arb_t s, slong prec);
void gamma_expectile(arb_t res, const arb_t quantile, const arb_t s, slong prec);
void gamma_quantile(arb_t res, slong k, slong n, const arb_t s, slong prec);
void normalized_median_gamma_rates(
        arb_struct *rates, slong n, const arb_t s, slong prec);

#ifdef __cplusplus
}
#endif

#endif
