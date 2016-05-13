#include "evaluate_site_frechet.h"
#include "util.h"

/*
 * This function computes expectations on edges according to the
 * conditional joint distribution of states at endpoints of the edge.
 * It's used for 'dwell' and 'trans' likelihood related calculations.
 * The marginal node vectors and the lhood node and edge vectors
 * will have already been evaluated.
 */
void
evaluate_site_frechet(
        arb_struct *edge_expectations,
        arb_mat_struct *marginal_node_vectors,
        arb_mat_struct *lhood_node_vectors,
        arb_mat_struct *lhood_edge_vectors,
        arb_mat_struct *frechet_matrices,
        csr_graph_t g, int *preorder,
        int node_count, int state_count, slong prec)
{
    int u, a, b;
    int idx;
    int start, stop;
    slong state;
    arb_mat_struct *lvec, *mvec, *evec;
    arb_mat_t fvec;
    arb_t tmp;

    arb_mat_init(fvec, state_count, 1);
    arb_init(tmp);

    for (u = 0; u < node_count; u++)
    {
        a = preorder[u];
        mvec = marginal_node_vectors + a;
        start = g->indptr[a];
        stop = g->indptr[a+1];

        for (idx = start; idx < stop; idx++)
        {
            b = g->indices[idx];
            /*
             * At this point (a, b) is an edge from node a to node b
             * in a pre-order traversal of edges of the tree.
             */
            lvec = lhood_node_vectors + b;
            evec = lhood_edge_vectors + idx;
            arb_mat_mul(fvec, frechet_matrices + idx, lvec, prec);

            arb_zero(edge_expectations + idx);
            for (state = 0; state < state_count; state++)
            {
                /* See arbplfmarginal.c regarding the zero case. */
                if (!arb_is_zero(arb_mat_entry(evec, state, 0)))
                {
                    arb_div(tmp,
                            arb_mat_entry(fvec, state, 0),
                            arb_mat_entry(evec, state, 0), prec);
                    arb_addmul(
                            edge_expectations + idx,
                            arb_mat_entry(mvec, state, 0),
                            tmp, prec);
                }
            }
        }
    }

    arb_clear(tmp);
    arb_mat_clear(fvec); 
}
