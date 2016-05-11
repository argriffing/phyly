#ifndef EVALUATE_SITE_FRECHET_H
#define EVALUATE_SITE_FRECHET_H

#include "flint/flint.h"
#include "arb.h"
#include "arb_mat.h"
#include "csr_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

void evaluate_site_frechet(
        arb_struct *edge_expectations,
        arb_mat_struct *marginal_node_vectors,
        arb_mat_struct *lhood_node_vectors,
        arb_mat_struct *lhood_edge_vectors,
        arb_mat_struct *frechet_matrices,
        csr_graph_t g, int *preorder,
        int node_count, int state_count, slong prec);

#ifdef __cplusplus
}
#endif

#endif
