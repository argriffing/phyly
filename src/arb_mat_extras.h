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

#ifdef __cplusplus
}
#endif

#endif
