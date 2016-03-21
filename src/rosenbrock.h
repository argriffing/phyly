#ifndef ROSENBROCK_H
#define ROSENBROCK_H


#include "arb.h"
#include "arb_mat.h"


#ifdef __cplusplus
extern "C" {
#endif

void rosenbrock_gradient(arb_t dx, arb_t dy,
        const arb_t x, const arb_t y, slong prec);

void rosenbrock_hessian(arb_mat_t H,
        const arb_t x, const arb_t y, slong prec);


#ifdef __cplusplus
}
#endif

#endif
