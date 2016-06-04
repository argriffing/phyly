#ifndef UTIL_H
#define UTIL_H

#include "flint/flint.h"

#include "arb.h"
#include "arb_mat.h"

#include "csr_graph.h"


#ifdef __cplusplus
extern "C" {
#endif

void _arb_sum(arb_t dest, arb_struct *src, slong len, slong prec);
int _arb_is_indeterminate(const arb_t x);
void _arb_set_si_2exp_si(arb_t x, slong man, slong exp);
void _arb_init_set(arb_t dest, const arb_t src);

void _arb_vec_printd(arb_srcptr vec, slong len, slong digits);

int _can_round(arb_t x);
double _arb_get_d(const arb_t x);
int _arb_vec_can_round(arb_struct * x, slong n);
int _arb_mat_can_round(arb_mat_t A);
void _arb_mat_scalar_div_d(arb_mat_t m, double d, slong prec);

int _arb_mat_solve_arb_vec(arb_struct *x,
        const arb_mat_t A, const arb_struct *b, slong prec);
void _arb_mat_indeterminate(arb_mat_t m);
int _arb_mat_is_indeterminate(const arb_mat_t m);
void _arb_mat_row_sums(arb_struct *dest, arb_mat_t src, slong prec);
void _arb_mat_div_entrywise(arb_mat_t c, arb_mat_t a, arb_mat_t b, slong prec);
void _arb_mat_ones(arb_mat_t A);
void _arb_mat_zero_diagonal(arb_mat_t A);
void _prune_update_rate(arb_mat_t d,
        const arb_mat_t c, const arb_mat_t a, const arb_mat_t b, slong prec);
void _prune_update_prob(arb_mat_t d,
        const arb_mat_t c, const arb_mat_t a, const arb_mat_t b, slong prec);
void _csr_graph_get_backward_maps(
        int *idx_to_a, int *b_to_idx, const csr_graph_t g);
void _csr_graph_get_preorder_edges(
        int *pre_to_idx, const csr_graph_t g, const int *preorder_nodes);
void _arb_update_rate_matrix_diagonal(arb_mat_t A, slong prec);

void _arb_vec_mul_arb_mat(
        arb_struct *z, const arb_struct *x, const arb_mat_t y, slong prec);
void _arb_mat_mul_AT_B(arb_mat_t C, const arb_mat_t A, const arb_mat_t B, slong prec);
void _arb_mat_exp_frechet(arb_mat_t P, arb_mat_t F,
        const arb_mat_t Q, const arb_mat_t L, slong prec);

void _expand_lower_triangular(arb_mat_t B, const arb_mat_t L);
void _arb_mat_proportions(arb_mat_t b, const arb_mat_t a, slong prec);

#ifdef __cplusplus
}
#endif

#endif
