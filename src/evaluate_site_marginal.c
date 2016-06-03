#include "evaluate_site_marginal.h"
#include "evaluate_site_forward.h"
#include "arb_mat_extras.h"
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
            if (arb_contains_zero(arb_mat_entry(b, i, j)))
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
                    /* debug */
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


/* fixme: deprecated */
static void
old_evaluate_site_marginal(
        arb_mat_struct *marginal_node_vectors,
        arb_mat_struct *base_node_vectors,
        arb_mat_struct *lhood_node_vectors,
        arb_mat_struct *lhood_edge_vectors,
        const root_prior_t r, const arb_struct *equilibrium,
        const arb_mat_struct *transition_matrices,
        csr_graph_struct *g, const navigation_t nav,
        int node_count, int state_count, slong prec)
{
    int u, a, b;
    int idx;
    int start, stop;
    arb_mat_struct *lvec, *mvec, *mvecb, *evec;
    const arb_mat_struct *tmat;
    arb_mat_t tmp;

    arb_mat_init(tmp, state_count, 1);

    /* Initialize using the root prior. */
    mvec = marginal_node_vectors + nav->preorder[0];
    _arb_mat_ones(mvec);
    root_prior_mul_col_vec(mvec, r, equilibrium, prec);

    for (u = 0; u < node_count; u++)
    {
        a = nav->preorder[u];
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


void
evaluate_site_marginal(
        arb_mat_struct *marginal_node_vectors,
        arb_mat_struct *base_node_vectors,
        arb_mat_struct *lhood_node_vectors,
        arb_mat_struct *lhood_edge_vectors,
        const root_prior_t r, const arb_struct *equilibrium,
        const arb_mat_struct *transition_matrices,
        csr_graph_struct *g, const navigation_t nav,
        int node_count, int state_count, slong prec)
{
    arb_t lhood;

    /* compute lhood */
    {
        const arb_mat_struct *lvec = lhood_node_vectors + nav->preorder[0];
        arb_init(lhood);
        root_prior_expectation(lhood, r, lvec, equilibrium, prec);
    }

    /* evaluate unnormalized marginal vectors */
    evaluate_site_marginal_unnormalized(
            marginal_node_vectors, base_node_vectors,
            lhood_node_vectors, lhood_edge_vectors,
            r, equilibrium, transition_matrices,
            g, nav, node_count, state_count, prec);

    /* divide each vector by the likelihood */
    {
        slong a;
        for (a = 0; a < node_count; a++)
        {
            arb_mat_struct *mvec = marginal_node_vectors + a;
            arb_mat_scalar_div_arb(mvec, mvec, lhood, prec);
        }
    }

    arb_clear(lhood);
}


void
evaluate_site_marginal_unnormalized(
        arb_mat_struct *marginal_node_vectors,
        arb_mat_struct *base_node_vectors,
        arb_mat_struct *lhood_node_vectors,
        arb_mat_struct *lhood_edge_vectors,
        const root_prior_t r, const arb_struct *equilibrium,
        const arb_mat_struct *transition_matrices,
        csr_graph_struct *g, const navigation_t nav,
        int node_count, int state_count, slong prec)
{
    /* Inefficiently allocate and compute forward vectors. */
    slong edge_count = node_count - 1;
    arb_mat_struct *forward_edge_vectors = _arb_mat_vec_init(
            state_count, 1, edge_count);
    arb_mat_struct *forward_node_vectors = _arb_mat_vec_init(
            state_count, 1, node_count);

    /* Evaluate forward edge vectors. */
    evaluate_site_forward(
            forward_edge_vectors, forward_node_vectors,
            base_node_vectors, lhood_edge_vectors,
            r, equilibrium, transition_matrices,
            g, nav, node_count, state_count, prec);

    /* Posterior decoding without normalization. */
    {
        slong u;
        for (u = 0; u < node_count; u++)
        {
            slong a = nav->preorder[u];
            _arb_mat_mul_entrywise(
                    marginal_node_vectors + a,
                    forward_node_vectors + a,
                    lhood_node_vectors + a, prec);
        }
    }

    _arb_mat_vec_clear(forward_edge_vectors, edge_count);
    _arb_mat_vec_clear(forward_node_vectors, node_count);
}
