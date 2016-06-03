#include "evaluate_site_marginal.h"
#include "evaluate_site_forward.h"
#include "arb_mat_extras.h"
#include "util.h"

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

    /* Inefficiently allocate and compute forward vectors. */
    slong edge_count = node_count - 1;
    arb_mat_struct *forward_edge_vectors = _arb_mat_vec_init(
            state_count, 1, edge_count);
    arb_mat_struct *forward_node_vectors = _arb_mat_vec_init(
            state_count, 1, node_count);

    /* Evaluate forward edge vectors. */
    evaluate_site_forward(forward_edge_vectors, forward_node_vectors,
            base_node_vectors, lhood_edge_vectors,
            r, equilibrium, transition_matrices,
            g, nav, node_count, state_count, prec);

    /* compute lhood */
    {
        const arb_mat_struct *lvec = lhood_node_vectors + nav->preorder[0];
        arb_init(lhood);
        root_prior_expectation(lhood, r, lvec, equilibrium, prec);
    }

    /* evaluate unnormalized marginal vectors */
    evaluate_site_marginal_unnormalized(marginal_node_vectors,
            forward_node_vectors, lhood_node_vectors, node_count, prec);

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
    _arb_mat_vec_clear(forward_edge_vectors, edge_count);
    _arb_mat_vec_clear(forward_node_vectors, node_count);
}


/* Posterior decoding without normalization. */
void
evaluate_site_marginal_unnormalized(
        arb_mat_struct *lhood_scaled_marginal_vectors,
        const arb_mat_struct *forward_node_vectors,
        const arb_mat_struct *lhood_node_vectors,
        int node_count, slong prec)
{
    slong a;
    for (a = 0; a < node_count; a++)
    {
        arb_mat_mul_entrywise(
                lhood_scaled_marginal_vectors + a,
                forward_node_vectors + a,
                lhood_node_vectors + a, prec);
    }
}
