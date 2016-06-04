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


void evaluate_site_marginal_unnormalized(
        arb_mat_struct *lhood_scaled_marginal_vectors,
        const arb_mat_struct *forward_node_vectors,
        const arb_mat_struct *lhood_node_vectors,
        int node_count, slong prec);


#ifdef __cplusplus
}
#endif

#endif
