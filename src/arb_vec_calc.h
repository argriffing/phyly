#ifndef ARB_VEC_CALC_H
#define ARB_VEC_CALC_H

#include "flint/flint.h"

#include "arf.h"
#include "arb.h"
#include "arb_mat.h"

#include "arb_vec_extras.h"


#ifdef __cplusplus
extern "C" {
#endif

/* extern TLS_PREFIX int arb_vec_calc_verbose; */

/*
 * `param` is the user data.
 * `n` is the dimensionality of the Euclidean space.
 */
typedef int (*arb_vec_calc_func_t) (
        arb_struct *vec_out, arb_mat_struct *jac_out,
        const arb_struct *inp, void *param, slong n, slong prec);

/*
 * Newton iteration.
 * Notation follows arb_calc_newton_step and arb_calc_refine_root_newton.
 */

int _arb_vec_calc_newton_delta(
        arb_struct *delta, arb_vec_calc_func_t func, void *param,
        const arb_struct *inp, slong n, slong prec);

int _arb_vec_calc_newton_step(
        arb_struct *xnew, arb_vec_calc_func_t func, void *param,
        const arb_struct *inp, slong n, slong prec);

int _arb_vec_calc_newton_contraction(
        arb_struct *xnew, arb_vec_calc_func_t func, void *param,
        const arb_struct *inp, slong n, slong prec);

int _arb_vec_calc_refine_root_newton(
        arb_struct *out, arb_vec_calc_func_t func, void *param,
        const arb_struct *start, slong n,
        slong eval_extra_prec, slong prec);

int _arb_vec_calc_krawczyk_contraction(
        arb_struct *xout, arb_vec_calc_func_t func, void *param,
        const arb_struct *xin, slong n, slong prec);

int _arb_vec_calc_refine_root_krawczyk(
        arb_struct *out, arb_vec_calc_func_t func, void *param,
        const arb_struct *start, slong n,
        slong eval_extra_prec, slong prec);

int _arb_vec_calc_refine_root_newton_midpoint(
        arb_struct *x_out,
        arb_vec_calc_func_t func, void *param,
        const arb_struct *x_start,
        slong n, slong prec);

#ifdef __cplusplus
}
#endif

#endif
