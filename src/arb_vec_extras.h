#ifndef ARB_VEC_EXTRAS_H
#define ARB_VEC_EXTRAS_H

#include "flint/flint.h"

#include "arb.h"


#ifdef __cplusplus
extern "C" {
#endif

void _arb_vec_printd(arb_srcptr vec, slong len, slong digits);
void _arb_vec_print(arb_srcptr vec, slong len);
int _arb_vec_contains(const arb_struct *a, const arb_struct *b, slong n);
int _arb_vec_overlaps(const arb_struct *a, const arb_struct *b, slong n);
int _arb_vec_intersection(arb_struct *c,
        const arb_struct *a, const arb_struct *b, slong n, slong prec);

#ifdef __cplusplus
}
#endif

#endif
