#include "stdlib.h"
#include "stdio.h"

#include "arb.h"
#include "arb_mat.h"

#include "util.h"
#include "model.h"


void
model_and_data_init(model_and_data_t m)
{
    m->use_equilibrium_root_prior = 0;
    m->use_equilibrium_rate_divisor = 0;
    m->root_node_index = -1;
    m->preorder = NULL;
    m->edge_rate_coefficients = NULL;
    csr_graph_init(m->g);
    arb_mat_init(m->mat, 0, 0);
    pmat_pre_init(m->p);
    csr_edge_mapper_pre_init(m->edge_map);
    arb_init(m->rate_divisor);
    arb_one(m->rate_divisor);
}

void
model_and_data_clear(model_and_data_t m)
{
    free(m->preorder);
    free(m->edge_rate_coefficients);
    csr_graph_clear(m->g);
    arb_mat_clear(m->mat);
    pmat_clear(m->p);
    csr_edge_mapper_clear(m->edge_map);
    arb_clear(m->rate_divisor);
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
        const pmat_t p, slong site,
        int use_equilibrium_root_prior, arb_struct *equilibrium,
        slong root_node_index, slong prec)
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

    /* optionally multiply by an equilibrium at the root node */
    if (use_equilibrium_root_prior)
    {
        bvec = base_node_vectors + root_node_index;
        for (j = 0; j < state_count; j++)
        {
            arb_struct *p;
            p = arb_mat_entry(bvec, j, 0);
            arb_mul(p, p, equilibrium + j, prec);
        }
    }
}


/*
 * Does not include scaling by edge rate coefficients.
 */
void
_update_rate_matrix_and_equilibrium(
        arb_mat_t rate_matrix,
        arb_struct *equilibrium,
        arb_t rate_divisor,
        int use_equilibrium_root_prior,
        int use_equilibrium_rate_divisor,
        const arb_mat_t mat,
        slong prec)
{
    /* Initialize the unscaled arbitrary precision rate matrix. */
    arb_mat_set(rate_matrix, mat);
    _arb_mat_zero_diagonal(rate_matrix);

    /* Update equilibrium if requested. */
    if (use_equilibrium_root_prior || use_equilibrium_rate_divisor)
    {
        _arb_vec_rate_matrix_equilibrium(equilibrium, rate_matrix, prec);
    }
    
    /*
     * Optionally update the rate matrix divisor.
     * This is the dot product of the rate matrix row sums
     * and the equilibrium distribution.
     */
    if (use_equilibrium_rate_divisor)
    {
        slong n;
        arb_struct *row_sums;

        n = arb_mat_nrows(rate_matrix);
        row_sums = _arb_vec_init(n);
        _arb_mat_row_sums(row_sums, rate_matrix, prec);
        _arb_vec_dot(rate_divisor, row_sums, equilibrium, n, prec);
        _arb_vec_clear(row_sums, n);
    }

    arb_mat_scalar_div_arb(rate_matrix, rate_matrix, rate_divisor, prec);
}
