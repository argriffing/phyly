#include "flint/flint.h"

#include "arb.h"
#include "arf.h"

#include "util.h"

#define FULL_RELATIVE_PRECISION 53

void
_arb_sum(arb_t dest, arb_struct *src, slong len, slong prec)
{
    slong i;
    arb_zero(dest);
    for (i = 0; i < len; i++)
    {
        arb_add(dest, dest, src + i, prec);
    }
}

int _can_round(arb_t x)
{
    /* This cannot deal with values like -1 +/- 1e-1000000000000 */
    /* return arb_can_round_arf(x, 53, ARF_RND_NEAR); */
    return arb_rel_accuracy_bits(x) >= FULL_RELATIVE_PRECISION;
}

int
_arb_is_indeterminate(const arb_t x)
{
    return arf_is_nan(arb_midref(x));
}

void
_arb_set_si_2exp_si(arb_t x, slong man, slong exp)
{
    arf_set_si_2exp_si(arb_midref(x), man, exp);
    mag_zero(arb_radref(x));
}

void
_arb_init_set(arb_t dest, const arb_t src)
{
    arb_init(dest);
    arb_set(dest, src);
}

int
_arb_mat_solve_arb_vec(arb_struct *x,
        const arb_mat_t A, const arb_struct *b, slong prec)
{
    int invertible;
    slong i, n;
    arb_mat_t xm, bm;
    n = arb_mat_nrows(A);
    arb_mat_init(xm, n, 1);
    arb_mat_init(bm, n, 1);
    for (i = 0; i < n; i++)
        arb_set(arb_mat_entry(bm, i, 0), b + i);
    invertible = arb_mat_solve(xm, A, bm, prec);
    for (i = 0; i < n; i++)
        arb_set(x + i, arb_mat_entry(xm, i, 0));
    arb_mat_clear(xm);
    arb_mat_clear(bm);
    return invertible;
}

void
_arb_mat_ones(arb_mat_t A)
{
    slong i, j;
    for (i = 0; i < arb_mat_nrows(A); i++)
    {
        for (j = 0; j < arb_mat_ncols(A); j++)
        {
            arb_one(arb_mat_entry(A, i, j));
        }
    }
}

void
_arb_mat_zero_diagonal(arb_mat_t A)
{
    slong i;
    for (i = 0; i < arb_mat_nrows(A) && i < arb_mat_ncols(A); i++)
    {
        arb_zero(arb_mat_entry(A, i, i));
    }
}

int
_arb_vec_can_round(arb_struct * x, slong n)
{
    slong i;
    for (i = 0; i < n; i++)
    {
        if (!_can_round(x + i))
        {
            return 0;
        }
    }
    return 1;
}

int
_arb_mat_can_round(arb_mat_t A)
{
    slong i, j;
    for (i = 0; i < arb_mat_nrows(A); i++)
    {
        for (j = 0; j < arb_mat_ncols(A); j++)
        {
            if (!_can_round(arb_mat_entry(A, i, j)))
            {
                return 0;
            }
        }
    }
    return 1;
}


int
_arb_mat_is_indeterminate(const arb_mat_t m)
{
    slong i, j;
    for (i = 0; i < arb_mat_nrows(m); i++)
        for (j = 0; j < arb_mat_ncols(m); j++)
            if (_arb_is_indeterminate(arb_mat_entry(m, i, j)))
                return 1;
    return 0;
}


void
_arb_mat_scalar_div_d(arb_mat_t m, double d, slong prec)
{
    arb_t x;
    arb_init(x);
    arb_set_d(x, d);
    arb_mat_scalar_div_arb(m, m, x, prec);
    arb_clear(x);
}


void
_arb_mat_indeterminate(arb_mat_t m)
{
    slong i, j;
    for (i = 0; i < arb_mat_nrows(m); i++)
    {
        for (j = 0; j < arb_mat_ncols(m); j++)
        {
            arb_indeterminate(arb_mat_entry(m, i, j));
        }
    }
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
_arb_mat_div_entrywise(arb_mat_t c, arb_mat_t a, arb_mat_t b, slong prec)
{
    slong i, j, nr, nc;

    nr = arb_mat_nrows(a);
    nc = arb_mat_ncols(a);

    for (i = 0; i < nr; i++)
    {
        for (j = 0; j < nc; j++)
        {
            arb_div(arb_mat_entry(c, i, j),
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
_prune_update(arb_mat_t D, arb_mat_t C, arb_mat_t A, arb_mat_t B, slong prec)
{
    /*
     * D = C o (A * B)
     * Analogous to _arb_mat_addmul
     * except with entrywise product instead of entrywise addition.
     */
    arb_t accum;
    arb_init(accum);
    slong i, j, k;
    slong r, s, t;
    r = arb_mat_nrows(A);
    s = arb_mat_ncols(A);
    t = arb_mat_ncols(B);
    if ((arb_mat_nrows(B) != s) ||
        (arb_mat_nrows(D) != r) || (arb_mat_ncols(D) != t) ||
        (arb_mat_nrows(C) != r) || (arb_mat_ncols(C) != t))
    {
        flint_fprintf(stderr, "internal error: incompatible dimensions\n");
        abort();
    }
    if (D == A || D == B || C == A || C == B)
    {
        flint_fprintf(stderr, "internal error: unsupported aliasing\n");
        abort();
    }
    for (i = 0; i < r; i++)
    {
        for (j = 0; j < t; j++)
        {
            arb_zero(accum);
            for (k = 0; k < s; k++)
            {
                arb_addmul(accum,
                        arb_mat_entry(A, i, k),
                        arb_mat_entry(B, k, j), prec);
            }
            arb_mul(arb_mat_entry(D, i, j),
                    arb_mat_entry(C, i, j), accum, prec);
        }
    }
    arb_clear(accum);
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

void
_csr_graph_get_preorder_edges(
        int *pre_to_idx, const csr_graph_t g, const int *preorder_nodes)
{
    int i, idx;
    int u, a;
    int node_count;
    node_count = g->n;
    i = 0;
    for (u = 0; u < node_count; u++)
    {
        a = preorder_nodes[u];
        for (idx = g->indptr[a]; idx < g->indptr[a+1]; idx++)
        {
            pre_to_idx[i++] = idx;
        }
    }
}

void
_arb_mat_sum(arb_t dst, arb_mat_t src, slong prec)
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

void
_arb_mat_row_sums(arb_struct *dest, arb_mat_t src, slong prec)
{
    slong i, j, r, c;

    if (arb_mat_is_empty(src))
        return;

    r = arb_mat_nrows(src);
    c = arb_mat_ncols(src);

    for (i = 0; i < r; i++)
    {
        arb_zero(dest + i);
        for (j = 0; j < c; j++)
        {
            arb_add(dest + i, dest + i, arb_mat_entry(src, i, j), prec);
        }
    }
}

void
_arb_vec_mul_arb_mat(
        arb_struct *z, const arb_struct *x, const arb_mat_t y, slong prec)
{
    slong nr, nc;
    slong i, j;
    arb_struct *w;

    nr = arb_mat_nrows(y);
    nc = arb_mat_ncols(y);
    if (z == x)
    {
        w = _arb_vec_init(nc);
    }
    else
    {
        w = z;
    }
    for (j = 0; j < nc; j++)
    {
        arb_zero(w+j);
        for (i = 0; i < nr; i++)
        {
            arb_addmul(w+j, x+i, arb_mat_entry(y, i, j), prec);
        }
    }
    if (z == x)
    {
        _arb_vec_set(z, w, nc);
        _arb_vec_clear(w, nc);
    }
}

void
_arb_mat_mul_AT_B(arb_mat_t C, const arb_mat_t A, const arb_mat_t B, slong prec)
{
    slong i, j, k;
    slong r, s, t;
    r = arb_mat_ncols(A);
    s = arb_mat_nrows(A);
    t = arb_mat_ncols(B);
    if (arb_mat_nrows(B) != s ||
        arb_mat_nrows(C) != r ||
        arb_mat_ncols(C) != t)
    {
        flint_fprintf(stderr, "internal error: incompatible dimensions\n");
        abort();
    }
    if (C == A || C == B)
    {
        flint_fprintf(stderr, "internal error: unsupported aliasing\n");
        abort();
    }
    arb_mat_zero(C);
    for (i = 0; i < r; i++)
    {
        for (j = 0; j < t; j++)
        {
            for (k = 0; k < s; k++)
            {
                arb_addmul(
                        arb_mat_entry(C, i, j),
                        arb_mat_entry(A, k, i),
                        arb_mat_entry(B, j, k), prec);
            }
        }
    }
}

void
_arb_mat_exp_frechet(arb_mat_t P, arb_mat_t F,
        const arb_mat_t Q, const arb_mat_t L, slong prec)
{
    slong n, i, j;
    arb_mat_t M;

    n = arb_mat_nrows(Q);
    arb_mat_init(M, 2*n, 2*n);

    /* Copy Q and L to matrix blocks of M */
    arb_mat_zero(M);
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            arb_set(arb_mat_entry(M, i, j), arb_mat_entry(Q, i, j));
            arb_set(arb_mat_entry(M, n+i, n+j), arb_mat_entry(Q, i, j));
            arb_set(arb_mat_entry(M, i, n+j), arb_mat_entry(L, i, j));
        }
    }

    /*
    flint_printf("debug: exp frechet matrix:\n");
    arb_mat_printd(M, 15); flint_printf("\n");
    */

    /* Compute the matrix exponential of M */
    arb_mat_exp(M, M, prec);

    /* Copy matrix blocks from M to P and from M to F */
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            arb_set(arb_mat_entry(P, i, j), arb_mat_entry(M, i, j));
            arb_set(arb_mat_entry(F, i, j), arb_mat_entry(M, i, n+j));
        }
    }

    arb_mat_clear(M);
}

void
_expand_lower_triangular(arb_mat_t B, const arb_mat_t L)
{
    slong i, j;
    arb_mat_set(B, L);
    for (i = 0; i < arb_mat_nrows(B); i++)
        for (j = i+1; j < arb_mat_ncols(B); j++)
            arb_set(arb_mat_entry(B, i, j), arb_mat_entry(B, j, i));
}


void
_arb_mat_proportions(arb_mat_t b, const arb_mat_t a, slong prec)
{
    int i, j;
    int zero_count, nonzero_count, indeterminate_count;
    slong r, c, size;
    arb_struct *p;
    
    r = arb_mat_nrows(b);
    c = arb_mat_ncols(b);
    size = r * c;

    zero_count = 0;
    nonzero_count = 0;
    indeterminate_count = 0;
    for (i = 0; i < r; i++)
    {
        for (j = 0; j < c; j++)
        {
            p = arb_mat_entry(a, i, j);
            if (_arb_is_indeterminate(p))
                indeterminate_count++;
            else if (arb_is_zero(p))
                zero_count++;
            else if (!arb_contains_zero(p))
                nonzero_count++;
        }
    }
    if (indeterminate_count || zero_count == size)
    {
        _arb_mat_indeterminate(b);
    }
    else if (nonzero_count == 1 && zero_count == size - 1)
    {
        for (i = 0; i < r; i++)
        {
            for (j = 0; j < c; j++)
            {
                p = arb_mat_entry(a, i, j);
                if (arb_is_zero(p))
                    arb_zero(arb_mat_entry(b, i, j));
                else
                    arb_one(arb_mat_entry(b, i, j));
            }
        }
    }
    else
    {
        arb_t total;
        arb_init(total);
        _arb_mat_sum(total, b, prec);
        arb_mat_scalar_div_arb(b, a, total, prec);
        arb_clear(total);
    }
}
