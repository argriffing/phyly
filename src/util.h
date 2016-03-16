#ifndef UTIL_H
#define UTIL_H

#include "flint/flint.h"

#include "arb.h"
#include "arb_mat.h"

#include "csr_graph.h"


#ifdef __cplusplus
extern "C" {
#endif

int _arb_is_indeterminate(const arb_t x);

void _arb_vec_printd(arb_srcptr vec, slong len, slong digits);

int _can_round(arb_t x);
int _arb_vec_can_round(arb_struct * x, slong n);
int _arb_mat_can_round(arb_mat_t A);

void _arb_mat_indeterminate(arb_mat_t m);
void _arb_mat_sum(arb_t dst, arb_mat_t src, slong prec);
void _arb_mat_row_sums(arb_struct *dest, arb_mat_t src, slong prec);
void _arb_mat_mul_entrywise(arb_mat_t c, arb_mat_t a, arb_mat_t b, slong prec);
void _arb_mat_div_entrywise(arb_mat_t c, arb_mat_t a, arb_mat_t b, slong prec);
void _arb_mat_ones(arb_mat_t A);
void _prune_update(arb_mat_t d, arb_mat_t c, arb_mat_t a, arb_mat_t b, slong prec);
void _csr_graph_get_backward_maps(int *idx_to_a, int *b_to_idx, csr_graph_t g);
void _arb_update_rate_matrix_diagonal(arb_mat_t A, slong prec);

void _arb_vec_mul_arb_mat(
        arb_struct *z, const arb_struct *x, const arb_mat_t y, slong prec);
void _arb_mat_mul_AT_B(arb_mat_t C, const arb_mat_t A, const arb_mat_t B, slong prec);
void _arb_mat_exp_frechet(arb_mat_t P, arb_mat_t F,
        const arb_mat_t Q, const arb_mat_t L, slong prec);

#ifdef __cplusplus
}
#endif

#endif
