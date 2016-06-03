/*
 * For each edge of the tree, compute a vector of likelihoods
 * for the part of the tree *above* the edge.
 * The incomplete/complete distinction refers to whether the
 * the transition along the edge has been included in the likelihoods.
 *
 * For the forward incomplete edge vectors,
 * each entry of the vector corresponds to a state at the node
 * at the top (rootward) end of the edge.
 * For the forward complete edge vectors,
 * each entry of the vector corresponds to a state at the node
 * at the bottom (leafward) end of the edge.
 *
 * The forward complete edge vectors are useful for computing
 * marginal state distributions (analogous to HMM posterior decoding)
 * and also for computing the forward incomplete edge vectors.
 * The forward incomplete edge vectors are useful for computing
 * conditional expectations of event histories on edges,
 * including linear combinations of state dwell proportions and
 * linear combinations of labeled state transition counts.
 */
#include "evaluate_site_forward.h"
#include "model.h"
#include "util.h"


void
evaluate_site_forward(
        arb_mat_struct *forward_incomplete_edge_vectors,
        arb_mat_struct *forward_complete_edge_vectors,
        arb_mat_struct *base_node_vectors,
        arb_mat_struct *lhood_edge_vectors,
        const root_prior_t r, const arb_struct *equilibrium,
        const arb_mat_struct *transition_matrices,
        csr_graph_struct *g, const navigation_t nav,
        int node_count, int state_count, slong prec)
{
    int u;
    int parent_idx, idx;
    int start, stop;
    arb_mat_t tmp;

    arb_mat_init(tmp, state_count, 1);

    for (u = 0; u < node_count; u++)
    {
        slong a = nav->preorder[u];

        /*
         * For each state at node 'a',
         * get the likelihood of everything above and including that node.
         */
        {
            const arb_mat_struct *bvec = base_node_vectors + a;
            if (u == 0)
            {
                _arb_mat_ones(tmp);
                root_prior_mul_col_vec(tmp, r, equilibrium, prec);
            }
            else
            {
                parent_idx = nav->b_to_idx[a];
                arb_mat_set(tmp, forward_complete_edge_vectors + parent_idx);
            }
            _arb_mat_mul_entrywise(tmp, tmp, bvec, prec);
        }

        start = g->indptr[a];
        stop = g->indptr[a+1];

        for (idx = start; idx < stop; idx++)
        {
            arb_mat_struct *fivec = forward_incomplete_edge_vectors + idx;
            arb_mat_struct *fcvec = forward_complete_edge_vectors + idx;

            /* compute forward incomplete edge vector of edge idx */
            {
                slong idx2;
                arb_mat_set(fivec, tmp);
                for (idx2 = start; idx2 < stop; idx2++)
                {
                    const arb_mat_struct *lvec = lhood_edge_vectors + idx2;
                    if (idx2 != idx)
                    {
                        _arb_mat_mul_entrywise(fivec, fivec, lvec, prec);
                    }
                }
            }

            /* compute forward complete edge vector of edge idx */
            {
                const arb_mat_struct *tmat = transition_matrices + idx;
                _arb_mat_mul_AT_B(fcvec, tmat, fivec, prec);
            }
        }
    }

    arb_mat_clear(tmp); 
}
