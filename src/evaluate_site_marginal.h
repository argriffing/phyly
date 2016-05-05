#ifndef EVALUATE_SITE_MARGINAL_H
#define EVALUATE_SITE_MARGINAL_H

#include "flint/flint.h"
#include "arb.h"
#include "arb_mat.h"
#include "csr_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

void evaluate_site_marginal(
        arb_mat_struct *marginal_node_vectors,
        arb_mat_struct *lhood_node_vectors,
        arb_mat_struct *lhood_edge_vectors,
        const arb_mat_struct *transition_matrices,
        csr_graph_struct *g, const int *preorder,
        int node_count, int state_count, slong prec);

#ifdef __cplusplus
}
#endif

#endif
