#include "evaluate_site_marginal.h"
#include "util.h"


static void
_arb_mat_div_entrywise_marginal(
        arb_mat_t c, const arb_mat_t a, const arb_mat_t b, slong prec)
{
    /*
     * The justification for 0/0 = 0 in this function is that
     * if the subtree likelihood conditional on a state is zero,
     * then it is OK if that state has zero marginal probability
     * at that node.
     */
    slong i, j, nr, nc;

    nr = arb_mat_nrows(a);
    nc = arb_mat_ncols(a);

    /*
    fprintf(stderr, "debug: dividing\n");
    arb_mat_printd(a, 15); flint_printf("\n");
    arb_mat_printd(b, 15); flint_printf("\n");
    */

    for (i = 0; i < nr; i++)
    {
        for (j = 0; j < nc; j++)
        {
            if (arb_is_zero(arb_mat_entry(b, i, j)))
            {
                if (arb_is_zero(arb_mat_entry(a, i, j)))
                {
                    /*
                    fprintf(stderr, "debug: 0/0 in marginal distribution\n");
                    */
                    arb_zero(arb_mat_entry(c, i, j));
                }
                else
                {
                    /*
                    fprintf(stderr, "internal error: unexpected ratio\n");
                    arb_mat_printd(a, 15); flint_printf("\n");
                    arb_mat_printd(b, 15); flint_printf("\n");
                    */
                    arb_indeterminate(arb_mat_entry(c, i, j));
                }
            }
            else
            {
                arb_div(arb_mat_entry(c, i, j),
                        arb_mat_entry(a, i, j),
                        arb_mat_entry(b, i, j), prec);
            }
        }
    }
}


void
evaluate_site_marginal(
        arb_mat_struct *marginal_node_vectors,
        arb_mat_struct *lhood_node_vectors,
        arb_mat_struct *lhood_edge_vectors,
        const arb_mat_struct *transition_matrices,
        csr_graph_struct *g, const int *preorder,
        int node_count, int state_count, slong prec)
{
    int u, a, b;
    int idx;
    int start, stop;
    arb_mat_struct *lvec, *mvec, *mvecb, *evec;
    const arb_mat_struct *tmat;
    arb_mat_t tmp;

    arb_mat_init(tmp, state_count, 1);

    _arb_mat_ones(marginal_node_vectors + preorder[0]);

    for (u = 0; u < node_count; u++)
    {
        a = preorder[u];
        lvec = lhood_node_vectors + a;
        mvec = marginal_node_vectors + a;
        start = g->indptr[a];
        stop = g->indptr[a+1];

        /*
         * Entrywise multiply by the likelihood node vector
         * and then normalize the distribution.
         */
        _arb_mat_mul_entrywise(mvec, mvec, lvec, prec);
        _arb_mat_proportions(mvec, mvec, prec);

        /*
        flint_printf("debug: mvec = \n");
        arb_mat_printd(mvec, 15); flint_printf("\n");
        */

        /* initialize neighboring downstream marginal vectors */
        for (idx = start; idx < stop; idx++)
        {
            b = g->indices[idx];
            /*
             * At this point (a, b) is an edge from node a to node b
             * in a pre-order traversal of edges of the tree.
             */
            evec = lhood_edge_vectors + idx;
            mvecb = marginal_node_vectors + b;
            tmat = transition_matrices + idx;

            /* todo: look into rewriting the dynamic programming to
             *       avoid this potentially destabilizing division
             *       while maintaining efficiency
             */
            _arb_mat_div_entrywise_marginal(tmp, mvec, evec, prec);
            _arb_mat_mul_AT_B(mvecb, tmat, tmp, prec);
        }
    }

    arb_mat_clear(tmp); 
}
