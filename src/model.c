#include "stdlib.h"
#include "stdio.h"

#include "arb.h"
#include "arb_mat.h"

#include "model.h"


void
model_and_data_init(model_and_data_t m)
{
    m->rate_divisor = 1.0;
    m->root_node_index = -1;
    m->preorder = NULL;
    m->edge_rate_coefficients = NULL;
    csr_graph_init(m->g);
    dmat_pre_init(m->mat);
    pmat_pre_init(m->p);
    csr_edge_mapper_pre_init(m->edge_map);
}

void
model_and_data_clear(model_and_data_t m)
{
    free(m->preorder);
    free(m->edge_rate_coefficients);
    csr_graph_clear(m->g);
    dmat_clear(m->mat);
    pmat_clear(m->p);
    csr_edge_mapper_clear(m->edge_map);
}



int
dmat_nrows(const dmat_t mat)
{
    return mat->r;
}

int
dmat_ncols(const dmat_t mat)
{
    return mat->c;
}

double *
dmat_entry(dmat_t mat, int i, int j)
{
    return mat->data + i * mat->c + j;
}

const double *
dmat_srcentry(const dmat_t mat, int i, int j)
{
    return mat->data + i * mat->c + j;
}

void
dmat_pre_init(dmat_t mat)
{
    /* either init or clear may follow this call */
    mat->data = NULL;
    mat->r = 0;
    mat->c = 0;
}

void
dmat_init(dmat_t mat, int r, int c)
{
    mat->data = malloc(r*c*sizeof(double));
    mat->r = r;
    mat->c = c;
}

void
dmat_clear(dmat_t mat)
{
    free(mat->data);
}

void
dmat_get_arb_mat(arb_mat_t dst, const dmat_t src)
{
    int i, j;
    int r, c;
    r = dmat_nrows(src);
    c = dmat_ncols(src);
    if (r != arb_mat_nrows(dst) || c != arb_mat_ncols(dst))
    {
        fprintf(stderr, "internal error: matrix size mismatch\n");
        abort();
    }
    for (i = 0; i < r; i++)
    {
        for (j = 0; j < c; j++)
        {
            arb_set_d(arb_mat_entry(dst, i, j), *dmat_srcentry(src, i, j));
        }
    }
}




int
pmat_nsites(const pmat_t mat)
{
    return mat->s;
}

int
pmat_nrows(const pmat_t mat)
{
    return mat->r;
}

int
pmat_ncols(const pmat_t mat)
{
    return mat->c;
}

double *
pmat_entry(pmat_t mat, int i, int j, int k)
{
    return mat->data + i * mat->r * mat->c + j * mat->c + k;
}

const double *
pmat_srcentry(const pmat_t mat, int i, int j, int k)
{
    return mat->data + i * mat->r * mat->c + j * mat->c + k;
}

void
pmat_pre_init(pmat_t mat)
{
    /* either init or clear may follow this call */
    mat->data = NULL;
    mat->s = 0;
    mat->r = 0;
    mat->c = 0;
}

void
pmat_init(pmat_t mat, int s, int r, int c)
{
    mat->data = malloc(s*r*c*sizeof(double));
    mat->s = s;
    mat->r = r;
    mat->c = c;
}

void
pmat_clear(pmat_t mat)
{
    free(mat->data);
}

void
pmat_update_base_node_vectors(
        arb_mat_struct *base_node_vectors,
        const pmat_t p, slong site)
{
    slong i, j;
    slong node_count, state_count;
    arb_mat_struct *bvec;
    node_count = pmat_nrows(p);
    state_count = pmat_ncols(p);
    for (i = 0; i < node_count; i++)
    {
        bvec = base_node_vectors + i;
        for (j = 0; j < state_count; j++)
        {
            arb_set_d(arb_mat_entry(bvec, j, 0), *pmat_srcentry(p, site, i, j));
        }
    }
}
