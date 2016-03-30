#ifndef EQUILIBRIUM_H
#define EQUILIBRIUM_H

#include "arb.h"
#include "arb_mat.h"

#ifdef __cplusplus
extern "C" {
#endif

void _arb_vec_rate_matrix_equilibrium(
        arb_struct *p, const arb_mat_t Q, slong prec);

void _arb_mat_rate_matrix_equilibrium(
        arb_mat_t p, const arb_mat_t Q, slong prec);


#ifdef __cplusplus
}
#endif

#endif
