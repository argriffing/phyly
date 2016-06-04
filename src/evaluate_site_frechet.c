#include "evaluate_site_frechet.h"
#include "util.h"

void
evaluate_site_frechet(
        arb_struct *lhood_scaled_edge_expectations,
        const arb_mat_struct *lhood_node_vectors,
        const arb_mat_struct *forward_edge_vectors,
        const arb_mat_struct *frechet_matrices,
        const csr_graph_t g, int *preorder,
        int node_count, int state_count, slong prec)
{
    slong u, idx, state;
    arb_mat_t fvec;

    arb_mat_init(fvec, state_count, 1);

    for (u = 0; u < node_count; u++)
    {
        slong a = preorder[u];
        slong start = g->indptr[a];
        slong stop = g->indptr[a+1];

        for (idx = start; idx < stop; idx++)
        {
            slong b = g->indices[idx];
            const arb_mat_struct *lvec = lhood_node_vectors + b;
            const arb_mat_struct *evec = forward_edge_vectors + idx;

            arb_zero(lhood_scaled_edge_expectations + idx);
            arb_mat_mul(fvec, frechet_matrices + idx, lvec, prec);
            for (state = 0; state < state_count; state++)
            {
                arb_addmul(lhood_scaled_edge_expectations + idx,
                        arb_mat_entry(fvec, state, 0),
                        arb_mat_entry(evec, state, 0), prec);
            }
        }
    }

    arb_mat_clear(fvec);
}
