#include "flint.h"

#include "util.h"

#include "arb.h"
#include "arf.h"


int _can_round(arb_t x)
{
    /* This cannot deal with values like -1 +/- 1e-1000000000000 */
    /* return arb_can_round_arf(x, 53, ARF_RND_NEAR); */
    return arb_rel_accuracy_bits(x) >= 53;
}

void
_arb_mat_mul_entrywise(arb_mat_t c, arb_mat_t a, arb_mat_t b, slong prec)
{
    slong i, j, nr, nc;

    nr = arb_mat_nrows(a);
    nc = arb_mat_ncols(a);

    for (i = 0; i < nr; i++)
    {
        for (j = 0; j < nc; j++)
        {
            arb_mul(arb_mat_entry(c, i, j),
                    arb_mat_entry(a, i, j),
                    arb_mat_entry(b, i, j), prec);
        }
    }
}

void
_arb_update_rate_matrix_diagonal(arb_mat_t A, slong prec)
{
    slong i, j;
    slong r, c;
    arb_ptr p;
    r = arb_mat_nrows(A);
    c = arb_mat_ncols(A);
    if (r != c)
    {
        flint_fprintf(stderr, "internal error: matrix must be square\n");
        abort();
    }
    for (i = 0; i < r; i++)
    {
        p = arb_mat_entry(A, i, i);
        arb_zero(p);
        for (j = 0; j < c; j++)
        {
            if (i != j)
            {
                arb_sub(p, p, arb_mat_entry(A, i, j), prec);
            }
        }
    }
}


void
_prune_update(arb_mat_t d, arb_mat_t c, arb_mat_t a, arb_mat_t b, slong prec)
{
    /*
     * d = c o a*b
     * Analogous to _arb_mat_addmul
     * except with entrywise product instead of entrywise addition.
     */
    slong m, n;

    m = a->r;
    n = b->c;

    arb_mat_t tmp;
    arb_mat_init(tmp, m, n);

    arb_mat_mul(tmp, a, b, prec);
    _arb_mat_mul_entrywise(d, c, tmp, prec);
    arb_mat_clear(tmp);
}

/*
 * Look at edges of the csr graph of the phylogenetic tree,
 * tracking edge_index->initial_node_index and final_node_index->edge_index.
 * Note that the edges do not need to be traversed in any particular order.
 */
void
_csr_graph_get_backward_maps(int *idx_to_a, int *b_to_idx, csr_graph_t g)
{
    int idx;
    int a, b;
    int node_count;
    node_count = g->n;
    for (b = 0; b < node_count; b++)
    {
        b_to_idx[b] = -1;
    }
    for (a = 0; a < node_count; a++)
    {
        for (idx = g->indptr[a]; idx < g->indptr[a+1]; idx++)
        {
            b = g->indices[idx];
            idx_to_a[idx] = a;
            b_to_idx[b] = idx;
        }
    }
}

void _arb_mat_sum(arb_t dst, arb_mat_t src, slong prec)
{
    slong i, j, r, c;
    r = arb_mat_nrows(src);
    c = arb_mat_ncols(src);
    arb_zero(dst);
    for (i = 0; i < r; i++)
    {
        for (j = 0; j < c; j++)
        {
            arb_add(dst, dst, arb_mat_entry(src, i, j), prec);
        }
    }
}
