#include "arb_mat.h"
#include "evaluate_site_forward.h"
#include "evaluate_site_marginal.h"

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
