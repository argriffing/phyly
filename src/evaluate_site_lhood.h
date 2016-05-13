#ifndef EVALUATE_SITE_LHOOD_H
#define EVALUATE_SITE_LHOOD_H

#include "flint/flint.h"
#include "arb.h"
#include "arb_mat.h"
#include "csr_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

void evaluate_site_lhood(arb_t lhood,
        arb_mat_struct *lhood_node_vectors,
        arb_mat_struct *lhood_edge_vectors,
        const arb_mat_struct *base_node_vectors,
        const arb_mat_struct *transition_matrices,
        const csr_graph_struct *g,
        const int *preorder, int node_count, slong prec);

#ifdef __cplusplus
}
#endif

#endif
