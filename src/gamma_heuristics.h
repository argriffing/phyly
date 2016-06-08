#ifndef GAMMA_HEURISTICS_H
#define GAMMA_HEURISTICS_H

#include "flint/flint.h"
#include "arb.h"


#ifdef __cplusplus
extern "C" {
#endif

/*
 * Note that any potential bugs in this functions would not affect
 * the correctness of the downstream gamma quantile calculations.
 * They would only affect the calculation efficiency.
 *
 * Eventually this functionality should be incorporated into arb.
 */

int gamma_quantile_temme_3_3(
        arb_t res, const arb_t s, const arb_t p, slong prec);

#ifdef __cplusplus
}
#endif

#endif

