#ifndef EVALUATE_SITE_MARGINAL_H
#define EVALUATE_SITE_MARGINAL_H

#include "flint/flint.h"
#include "arb.h"
#include "arb_mat.h"
#include "csr_graph.h"
#include "model.h"

#ifdef __cplusplus
extern "C" {
#endif

evaluate_site_forward(
        arb_mat_struct *forward_incomplete_edge_vectors,
        arb_mat_struct *forward_complete_edge_vectors,
        arb_mat_struct *base_node_vectors,
        arb_mat_struct *lhood_edge_vectors,
        const root_prior_t r, const arb_struct *equilibrium,
        const arb_mat_struct *transition_matrices,
        csr_graph_struct *g, const int *preorder,
        const int *idx_to_a, const int *b_to_idx,
        int node_count, int state_count, slong prec)

#ifdef __cplusplus
}
#endif

#endif
