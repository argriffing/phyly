#include "stdlib.h"
#include "stdio.h"

#include "arb.h"
#include "arb_mat.h"

#include "util.h"
#include "equilibrium.h"
#include "arb_mat_extras.h"
#include "model.h"


void
model_and_data_init(model_and_data_t m)
{
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
    root_prior_pre_init(m->root_prior);
    rate_mixture_pre_init(m->rate_mixture);
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
    root_prior_clear(m->root_prior);
    rate_mixture_clear(m->rate_mixture);
}

slong
model_and_data_edge_count(const model_and_data_t m)
{
    return m->g->nnz;
}

slong
model_and_data_node_count(const model_and_data_t m)
{
    return m->g->n;
}

slong
model_and_data_state_count(const model_and_data_t m)
{
    return arb_mat_nrows(m->mat);
}

slong
model_and_data_site_count(const model_and_data_t m)
{
    return pmat_nsites(m->p);
}

slong
model_and_data_rate_category_count(const model_and_data_t m)
{
    return rate_mixture_category_count(m->rate_mixture);
}

int
model_and_data_uses_equilibrium(const model_and_data_t m)
{
    return (m->root_prior->mode == ROOT_PRIOR_EQUILIBRIUM ||
            m->use_equilibrium_rate_divisor);
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

    /* Optionally update the root node vector. */
    /*
    bvec = base_node_vectors + root_node_index;
    root_prior_mul_col_vec(bvec, root_prior, equilibrium, prec);
    */
}



void
root_prior_pre_init(root_prior_t r)
{
    r->n = 0;
    r->mode = ROOT_PRIOR_UNDEFINED;
    r->custom_distribution = NULL;
}

void
root_prior_init(root_prior_t r, int n, enum root_prior_mode mode)
{
    r->n = n;
    r->mode = mode;
    r->custom_distribution = NULL;
}

void
root_prior_clear(root_prior_t r)
{
    free(r->custom_distribution);
}

void
root_prior_mul_col_vec(arb_mat_t A, const root_prior_t r,
        const arb_struct *equilibrium, slong prec)
{
    slong i;
    if (arb_mat_ncols(A) != 1)
    {
        flint_fprintf(stderr, "internal error (root_prior_expectation): "
                "expected 1 column but observed %wd\n", arb_mat_ncols(A));
        abort();
    }
    if (arb_mat_nrows(A) != r->n)
    {
        flint_fprintf(stderr, "internal error (root_prior_expectation): "
                "expected %d rows but observed %wd\n", r->n, arb_mat_nrows(A));
        abort();
    }
    if (r->mode == ROOT_PRIOR_NONE)
    {
        return;
    }
    else if (r->mode == ROOT_PRIOR_UNIFORM)
    {
        for (i = 0; i < r->n; i++)
        {
            arb_ptr p = arb_mat_entry(A, i, 0);
            arb_div_si(p, p, r->n, prec);
        }
    }
    else if (r->mode == ROOT_PRIOR_EQUILIBRIUM)
    {
        if (!equilibrium) abort(); /* assert */
        for (i = 0; i < r->n; i++)
        {
            arb_ptr p = arb_mat_entry(A, i, 0);
            arb_mul(p, p, equilibrium + i, prec);
        }
    }
    else if (r->mode == ROOT_PRIOR_CUSTOM)
    {
        arb_t d;
        arb_init(d);
        if (!r->custom_distribution) abort(); /* assert */
        for (i = 0; i < r->n; i++)
        {
            arb_ptr p = arb_mat_entry(A, i, 0);
            arb_set_d(d, r->custom_distribution[i]);
            arb_mul(p, p, d, prec);
        }
        arb_clear(d);
    }
    else
    {
        abort(); /* assert */
    }
}

void
root_prior_expectation(arb_t out,
        const root_prior_t r, const arb_mat_t A,
        const arb_struct *equilibrium, slong prec)
{
    slong i;
    if (arb_mat_ncols(A) != 1)
    {
        flint_fprintf(stderr, "internal error (root_prior_expectation): "
                "expected 1 column but observed %wd\n", arb_mat_ncols(A));
        abort();
    }
    if (arb_mat_nrows(A) != r->n)
    {
        flint_fprintf(stderr, "internal error (root_prior_expectation): "
                "expected %d rows but observed %wd\n", r->n, arb_mat_nrows(A));
        abort();
    }
    if (r->mode == ROOT_PRIOR_NONE)
    {
        _arb_mat_sum(out, A, prec);
    }
    else if (r->mode == ROOT_PRIOR_UNIFORM)
    {
        if (_arb_mat_col_is_constant(A, 0))
        {
            arb_set(out, arb_mat_entry(A, 0, 0));
        }
        else
        {
            _arb_mat_sum(out, A, prec);
            arb_div_si(out, out, r->n, prec);
        }
    }
    else if (r->mode == ROOT_PRIOR_EQUILIBRIUM)
    {
        if (!equilibrium) abort(); /* assert */
        if (_arb_mat_col_is_constant(A, 0) &&
            _arb_vec_is_finite(equilibrium, r->n))
        {
            arb_set(out, arb_mat_entry(A, 0, 0));
        }
        else
        {
            arb_zero(out);
            for (i = 0; i < r->n; i++)
            {
                arb_addmul(out, equilibrium + i, arb_mat_entry(A, i, 0), prec);
            }
        }
    }
    else if (r->mode == ROOT_PRIOR_CUSTOM)
    {
        arb_t d;
        arb_init(d);
        if (!r->custom_distribution) abort(); /* assert */
        arb_zero(out);
        for (i = 0; i < r->n; i++)
        {
            arb_set_d(d, r->custom_distribution[i]);
            arb_addmul(out, arb_mat_entry(A, i, 0), d, prec);
        }
        arb_clear(d);
    }
    else
    {
        abort(); /* assert */
    }
}
