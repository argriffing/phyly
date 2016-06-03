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

/*
 * incomplete_edge_vectors -> edge_vectors
 * complete_edge_vectors -> node_vectors
 */

void
evaluate_site_forward(
        arb_mat_struct *forward_edge_vectors,
        arb_mat_struct *forward_node_vectors,
        arb_mat_struct *base_node_vectors,
        arb_mat_struct *lhood_edge_vectors,
        const root_prior_t r, const arb_struct *equilibrium,
        const arb_mat_struct *transition_matrices,
        csr_graph_struct *g, const navigation_t nav,
        int node_count, int state_count, slong prec)
{
    slong u;
    int idx;
    int start, stop;
    arb_mat_t tmp;

    /*
     * Define the forward node vector at the root.
     * The forward node vector is about the likelihood above,
     * but not including, the node of interest.
     */
    {
        slong a = nav->preorder[0];
        arb_mat_struct *fnvec = forward_node_vectors + a;
        _arb_mat_ones(fnvec);
        root_prior_mul_col_vec(fnvec, r, equilibrium, prec);
    }

    arb_mat_init(tmp, state_count, 1);

    for (u = 0; u < node_count; u++)
    {
        slong a = nav->preorder[u];

        /*
         * The tmp vector is about the likelihood above,
         * and including, the node of interest.
         */
        _arb_mat_mul_entrywise(tmp,
                forward_node_vectors + a,
                base_node_vectors + a, prec);

        start = g->indptr[a];
        stop = g->indptr[a+1];

        for (idx = start; idx < stop; idx++)
        {
            slong b = g->indices[idx];
            arb_mat_struct *fevec = forward_edge_vectors + idx;
            arb_mat_struct *fnvec = forward_node_vectors + b;

            /* compute forward edge vector of edge idx */
            {
                slong idx2;
                arb_mat_set(fevec, tmp);
                for (idx2 = start; idx2 < stop; idx2++)
                {
                    const arb_mat_struct *lvec = lhood_edge_vectors + idx2;
                    if (idx2 != idx)
                    {
                        _arb_mat_mul_entrywise(fevec, fevec, lvec, prec);
                    }
                }
            }

            /* compute forward node vector of node b */
            {
                const arb_mat_struct *tmat = transition_matrices + idx;
                _arb_mat_mul_AT_B(fnvec, tmat, fevec, prec);
            }
        }
    }

    arb_mat_clear(tmp); 
}
