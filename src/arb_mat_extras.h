#ifndef ARB_MAT_EXTRAS_H
#define ARB_MAT_EXTRAS_H

#include "flint/flint.h"

#include "arb.h"
#include "arb_mat.h"

#ifdef __cplusplus
extern "C" {
#endif

arb_mat_struct * _arb_mat_vec_init(slong nrows, slong ncols, slong len);
void _arb_mat_vec_clear(arb_mat_struct *vec, slong len);
int _arb_mat_col_is_constant(const arb_mat_t A, slong j);
void _arb_mat_mul_stochastic(
        arb_mat_t C, const arb_mat_t A, const arb_mat_t B, slong prec);
void _arb_mat_mul_rate_matrix(
        arb_mat_t C, const arb_mat_t A, const arb_mat_t B, slong prec);
int _arb_mat_is_finite(const arb_mat_t A);
void _arb_mat_sum(arb_t dst, const arb_mat_t src, slong prec);

#ifdef __cplusplus
}
#endif

#endif
