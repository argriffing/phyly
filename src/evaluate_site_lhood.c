#include "evaluate_site_lhood.h"
#include "util.h"

/* Calculate the likelihood, storing many intermediate calculations. */
void
evaluate_site_lhood(arb_t lhood,
        arb_mat_struct *lhood_node_vectors,
        arb_mat_struct *lhood_edge_vectors,
        arb_mat_struct *base_node_vectors,
        arb_mat_struct *transition_matrices,
        csr_graph_struct *g, int *preorder, int node_count, slong prec)
{
    int u, a, b, idx;
    int start, stop;
    arb_mat_struct *nmat, *nmatb, *tmat, *emat;

    /*
     * Fill all of the per-node and per-edge likelihood-related vectors.
     * Note that because edge derivatives are requested,
     * the vectors on edges are stored explicitly.
     * In the likelihood-only variant of this function, these per-edge
     * vectors are temporary variables whose lifespan is only long enough
     * to update the vector associated with the parent node of the edge.
     */
    for (u = 0; u < node_count; u++)
    {
        a = preorder[node_count - 1 - u];
        nmat = lhood_node_vectors + a;
        start = g->indptr[a];
        stop = g->indptr[a+1];

        /* initialize the state vector for node a */
        arb_mat_set(nmat, base_node_vectors + a);

        /* create all of the state vectors on edges outgoing from this node */
        for (idx = start; idx < stop; idx++)
        {
            b = g->indices[idx];
            /*
             * At this point (a, b) is an edge from node a to node b
             * in a post-order traversal of edges of the tree.
             */
            tmat = transition_matrices + idx;
            nmatb = lhood_node_vectors + b;
            /*
             * Update vectors on edges if requested,
             * otherwise avoid using temporary vectors on edges.
             * In either case, lhood node vectors are updated.
             */
            if (lhood_edge_vectors)
            {
                emat = lhood_edge_vectors + idx;
                arb_mat_mul(emat, tmat, nmatb, prec);
                _arb_mat_mul_entrywise(nmat, nmat, emat, prec);
            }
            else
            {
                _prune_update(nmat, nmat, tmat, nmatb, prec);
            }
        }
    }

    /* Report the sum of state entries associated with the root. */
    int root_node_index = preorder[0];
    nmat = lhood_node_vectors + root_node_index;
    _arb_mat_sum(lhood, nmat, prec);
}
