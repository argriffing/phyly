#include "stdlib.h"
#include "stdio.h"

#include "arb.h"
#include "arb_mat.h"

#include "util.h"
#include "equilibrium.h"
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
        const pmat_t p, slong site,
        const root_prior_t root_prior,
        const arb_struct *equilibrium,
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

    /* Optionally update the root node vector. */
    bvec = base_node_vectors + root_node_index;
    root_prior_mul_col_vec(bvec, root_prior, equilibrium, prec);
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
    if (arb_mat_ncols(A) != 1) abort(); /* assert */
    if (arb_mat_nrows(A) != r->n) abort(); /* assert */
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
rate_mixture_pre_init(rate_mixture_t x)
{
    x->n = 0;
    x->rates = NULL;
    x->prior = NULL;
    x->mode = RATE_MIXTURE_UNDEFINED;
}

void
rate_mixture_init(rate_mixture_t x, int n)
{
    x->n = n;
    x->rates = flint_malloc(n * sizeof(double));
    x->prior = flint_malloc(n * sizeof(double));
}

void
rate_mixture_clear(rate_mixture_t x)
{
    flint_free(x->rates);
    flint_free(x->prior);
    x->mode = RATE_MIXTURE_UNDEFINED;
}

void
rate_mixture_get_rate(arb_t rate, const rate_mixture_t x, slong idx)
{
    if (x->mode == RATE_MIXTURE_UNDEFINED)
    {
        flint_fprintf(stderr, "internal error: undefined rate mixture\n");
        abort();
    }
    else if (x->mode == RATE_MIXTURE_NONE)
    {
        arb_one(rate);
    }
    else if (x->mode == RATE_MIXTURE_UNIFORM || x->mode == RATE_MIXTURE_CUSTOM)
    {
        arb_set_d(rate, x->rates[idx]);
    }
    else
    {
        flint_fprintf(stderr, "internal error: "
                      "unrecognized rate mixture mode\n");
        abort();
    }
}

void
rate_mixture_get_prob(arb_t prob, const rate_mixture_t x,
        slong idx, slong prec)
{
    if (x->mode == RATE_MIXTURE_UNDEFINED)
    {
        flint_fprintf(stderr, "internal error: undefined rate mixture\n");
        abort();
    }
    else if (x->mode == RATE_MIXTURE_NONE)
    {
        arb_one(prob);
    }
    else if (x->mode == RATE_MIXTURE_UNIFORM)
    {
        /*
         * This code branch involves a division that could
         * unnecessarily lose exactness in some situations.
         */
        arb_set_si(prob, x->n);
        arb_inv(prob, prob, prec);
    }
    else if (x->mode == RATE_MIXTURE_CUSTOM)
    {
        arb_set_d(prob, x->prior[idx]);
    }
    else
    {
        flint_fprintf(stderr, "internal error: "
                      "unrecognized rate mixture mode\n");
        abort();
    }
}

slong
rate_mixture_category_count(const rate_mixture_t x)
{
    if (x->mode == RATE_MIXTURE_UNDEFINED)
    {
        flint_fprintf(stderr, "internal error: undefined rate mixture\n");
        abort();
    }
    else if (x->mode == RATE_MIXTURE_NONE)
    {
        return 1;
    }
    else if (x->mode == RATE_MIXTURE_UNIFORM || x->mode == RATE_MIXTURE_CUSTOM)
    {
        return x->n;
    }
    else
    {
        flint_fprintf(stderr, "internal error: "
                      "unrecognized rate mixture mode\n");
        abort();
    }
}

void
rate_mixture_expectation(arb_t rate, const rate_mixture_t x, slong prec)
{
    slong i;
    arb_t tmp, tmpb;
    if (x->mode == RATE_MIXTURE_UNDEFINED)
    {
        flint_fprintf(stderr, "internal error: undefined rate mixture\n");
        abort();
    }
    else if (x->mode == RATE_MIXTURE_NONE)
    {
        flint_fprintf(stderr, "internal error: computing the expected rate "
                      "of a trivial rate mixture\n");
        abort();
    }
    else if (x->mode == RATE_MIXTURE_UNIFORM || x->mode == RATE_MIXTURE_CUSTOM)
    {
        arb_init(tmp);
        arb_init(tmpb);
        arb_zero(rate);
        if (x->mode == RATE_MIXTURE_UNIFORM)
        {
            for (i = 0; i < x->n; i++)
            {
                arb_set_d(tmp, x->rates[i]);
                arb_add(rate, rate, tmp, prec);
            }
            arb_div_si(rate, rate, x->n, prec);
        }
        else if (x->mode == RATE_MIXTURE_CUSTOM)
        {
            for (i = 0; i < x->n; i++)
            {
                arb_set_d(tmp, x->rates[i]);
                arb_set_d(tmpb, x->prior[i]);
                arb_addmul(rate, tmp, tmpb, prec);
            }
        }
        arb_clear(tmp);
        arb_clear(tmpb);
    }
    else
    {
        flint_fprintf(stderr, "internal error: "
                      "unrecognized rate mixture mode\n");
        abort();
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
        int use_equilibrium_rate_divisor,
        const root_prior_t root_prior,
        const rate_mixture_t rate_mixture,
        const arb_mat_t mat,
        slong prec)
{
    /* Initialize the unscaled arbitrary precision rate matrix. */
    arb_mat_set(rate_matrix, mat);
    _arb_mat_zero_diagonal(rate_matrix);

    /* Update equilibrium if requested. */
    if (use_equilibrium_rate_divisor ||
        root_prior->mode == ROOT_PRIOR_EQUILIBRIUM)
    {
        _arb_vec_rate_matrix_equilibrium(equilibrium, rate_matrix, prec);
    }
    
    /*
     * Optionally update the rate matrix divisor.
     * This is the dot product of the rate matrix row sums
     * and the equilibrium distribution,
     * multiplied by the dot product of the mixture rates
     * and their stationary probabilities.
     */
    if (use_equilibrium_rate_divisor)
    {
        slong state_count;
        arb_struct *row_sums;

        state_count = arb_mat_nrows(rate_matrix);
        row_sums = _arb_vec_init(state_count);
        _arb_mat_row_sums(row_sums, rate_matrix, prec);
        _arb_vec_dot(rate_divisor, row_sums, equilibrium, state_count, prec);
        _arb_vec_clear(row_sums, state_count);
        if (rate_mixture->mode != RATE_MIXTURE_NONE)
        {
            arb_t tmp;
            arb_init(tmp);
            rate_mixture_expectation(tmp, rate_mixture, prec);
            arb_mul(rate_divisor, rate_divisor, tmp, prec);
            arb_clear(tmp);
        }
    }

    arb_mat_scalar_div_arb(rate_matrix, rate_matrix, rate_divisor, prec);
}
