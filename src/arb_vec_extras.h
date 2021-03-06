#ifndef ARB_VEC_EXTRAS_H
#define ARB_VEC_EXTRAS_H

#include "flint/flint.h"

#include "arb.h"


#ifdef __cplusplus
extern "C" {
#endif

int _arb_vec_is_nonnegative(arb_srcptr vec, slong len);
void _arb_vec_add_error_arb_vec(arb_ptr res, arb_srcptr err, slong len);
slong _arb_vec_min_rel_accuracy_bits(const arb_struct *v, slong n);
void _arb_vec_proportions(arb_struct *b, const arb_struct *a, slong n, slong prec);
void _arb_vec_mid(arb_struct *y, const arb_struct *x, slong n);
void _arb_vec_printd(arb_srcptr vec, slong len, slong digits);
void _arb_vec_print(arb_srcptr vec, slong len);
int _arb_vec_is_indeterminate(const arb_struct *v, slong n);
int _arb_vec_equal(const arb_struct *a, const arb_struct *b, slong n);
int _arb_vec_contains(const arb_struct *a, const arb_struct *b, slong n);
int _arb_vec_contains_zero(const arb_struct *a, slong n);
int _arb_vec_overlaps(const arb_struct *a, const arb_struct *b, slong n);
int _arb_vec_intersection(arb_struct *c,
        const arb_struct *a, const arb_struct *b, slong n, slong prec);
void _arb_vec_div(arb_struct *c,
        const arb_struct *a, const arb_struct *b, slong n, slong prec);
void _arb_vec_scalar_sub(arb_ptr res, arb_srcptr vec,
        slong len, const arb_t c, slong prec);
void _arb_vec_scalar_mul_si(arb_ptr res, arb_srcptr vec,
        slong len, slong y, slong prec);

#ifdef __cplusplus
}
#endif

#endif
