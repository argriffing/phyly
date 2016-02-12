#ifndef UTIL_H
#define UTIL_H

#include "flint.h"

#include "arb.h"
#include "arb_mat.h"

#include "csr_graph.h"


#ifdef __cplusplus
extern "C" {
#endif

int _can_round(arb_t x);
void _arb_mat_sum(arb_t dst, arb_mat_t src, slong prec);
void _arb_mat_mul_entrywise(arb_mat_t c, arb_mat_t a, arb_mat_t b, slong prec);
void _prune_update(arb_mat_t d, arb_mat_t c, arb_mat_t a, arb_mat_t b, slong prec);
void _csr_graph_get_backward_maps(int *idx_to_a, int *b_to_idx, csr_graph_t g);
void _arb_update_rate_matrix_diagonal(arb_mat_t A, slong prec);

#ifdef __cplusplus
}
#endif

#endif
