#ifndef EVALUATE_SITE_FORWARD_H
#define EVALUATE_SITE_FORWARD_H

#include "flint/flint.h"
#include "arb.h"
#include "arb_mat.h"
#include "csr_graph.h"
#include "model.h"

#ifdef __cplusplus
extern "C" {
#endif

void
evaluate_site_forward(
        arb_mat_struct *forward_edge_vectors,
        arb_mat_struct *forward_node_vectors,
        arb_mat_struct *base_node_vectors,
        arb_mat_struct *lhood_edge_vectors,
        const root_prior_t r, const arb_struct *equilibrium,
        const arb_mat_struct *transition_matrices,
        csr_graph_struct *g, const navigation_t nav,
        int node_count, int state_count, slong prec);

#ifdef __cplusplus
}
#endif

#endif
