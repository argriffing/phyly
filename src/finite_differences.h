#ifndef FINITE_DIFFERENCES_H
#define FINITE_DIFFERENCES_H

#include "flint/flint.h"

#include "arb.h"
#include "arb_mat.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef void (* multivariate_function_t) (
        arb_t y, const arb_struct *x, void *param,
        slong n, slong prec);

typedef void (* multivariate_vector_function_t) (
        arb_struct *y, const arb_struct *x, void *param,
        slong n, slong k, slong prec);


typedef struct
{
    multivariate_function_t func;
    void *param;
    arb_t delta;
} gradient_param_struct;
typedef gradient_param_struct gradient_param_t[1];

void gradient_param_init(gradient_param_t g,
        multivariate_function_t func, void *param, const arb_t delta);

void gradient_param_clear(gradient_param_t g);

void finite_differences_gradient(
        arb_struct *gradient,
        const arb_struct *x, void *param,
        slong n, slong k, slong prec);

void finite_differences_jacobian(arb_mat_t J,
        multivariate_vector_function_t func, void *param,
        const arb_struct *x, slong n, slong k, const arb_t delta, slong prec);

void finite_differences_hessian(arb_mat_t H,
        multivariate_function_t func, void *param,
        const arb_struct *x, slong n, const arb_t delta, slong prec);

#ifdef __cplusplus
}
#endif

#endif
