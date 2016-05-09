/*
 * Define a cross-site workspace.
 * The lifetime of the object does not extend across precision levels,
 * but it does extend across independent sites in the alignment.
 * The object may be destroyed and reconstructed at each iteration
 * across precision levels, but the object is constant across alignment sites.
 *
 * This is a 'middle layer.'
 * The outer layer is precision-agnostic and stores only the user input.
 * This middle layer depends on precision but is site-agnostic.
 * The inner layer depends on both the precision and the site.
 *
 * The code for the outer layer is already re-used across roughly a dozen
 * functions related to phylogenetic CTMC models; the code for the middle
 * layer is intended to be similarly re-used.
 */
#include "flint/flint.h"

#include "arb_mat.h"
#include "arb.h"

#include "csr_graph.h"
#include "model.h"
#include "equilibrium.h"
#include "util.h"
#include "arb_mat_extras.h"
#include "cross_site_ws.h"


void
tmat_collection_pre_init(tmat_collection_t x)
{
    x->rate_category_count = 0;
    x->edge_count = 0;
    x->matrices = NULL;
}

void
tmat_collection_init(tmat_collection_t x,
        slong rate_category_count, slong edge_count, slong state_count)
{
    slong n = rate_category_count * edge_count;
    x->rate_category_count = rate_category_count;
    x->edge_count = edge_count;
    x->matrices = _arb_mat_vec_init(state_count, state_count, n);
}

void
tmat_collection_clear(tmat_collection_t x)
{
    slong n = x->rate_category_count * x->edge_count;
    _arb_mat_vec_clear(x->matrices, n);
}

arb_mat_struct *
tmat_collection_entry(tmat_collection_t x,
        slong rate_category_idx, slong edge_idx)
{
    slong offset;
    offset = rate_category_idx * x->edge_count + edge_idx;
    return x->matrices + offset;
}




void
cross_site_ws_pre_init(cross_site_ws_t w)
{
    w->prec = 0;
    w->rate_category_count = 0;
    w->node_count = 0;
    w->edge_count = 0;
    w->state_count = 0;
    w->edge_rates = NULL;
    w->equilibrium = NULL;
    w->rate_divisor = NULL;
    w->rate_matrix = NULL;
    tmat_collection_pre_init(w->transition_matrices);
}

static void
_cross_site_ws_init_edge_rates(cross_site_ws_t w, model_and_data_t m)
{
    slong i, idx;
    double tmpd;

    w->edge_rates = _arb_vec_init(w->edge_count);

    /*
     * Define the map from csr edge index to edge rate.
     * The edge rate is represented in arbitrary precision.
     */
    if (!m->edge_map)
    {
        fprintf(stderr, "internal error: edge map is uninitialized\n");
        abort();
    }
    if (!m->edge_map->order)
    {
        fprintf(stderr, "internal error: edge map order is uninitialized\n");
        abort();
    }
    if (!m->edge_rate_coefficients)
    {
        fprintf(stderr, "internal error: edge rate coeffs unavailable\n");
        abort();
    }
    for (i = 0; i < w->edge_count; i++)
    {
        idx = m->edge_map->order[i];
        tmpd = m->edge_rate_coefficients[i];
        arb_set_d(w->edge_rates + idx, tmpd);
    }
}

void
cross_site_ws_init(cross_site_ws_t w, model_and_data_t m, slong prec)
{
    w->prec = prec;
    w->rate_category_count = model_and_data_rate_category_count(m);
    w->node_count = model_and_data_node_count(m);
    w->edge_count = model_and_data_edge_count(m);
    w->state_count = model_and_data_state_count(m);

    /* initialize edge rates */
    _cross_site_ws_init_edge_rates(w, m);

    /* initialize the equilibrium if necessary, ignoring diagonal entries */
    if (model_and_data_uses_equilibrium(m))
    {
        w->equilibrium = _arb_vec_init(w->state_count);
        _arb_vec_rate_matrix_equilibrium(
                w->equilibrium, m->mat, w->prec);
    }

    /* initialize the unscaled rate matrix, and zero the diagonal */
    {
        w->rate_matrix = flint_malloc(sizeof(arb_mat_struct));
        arb_mat_init(w->rate_matrix, w->state_count, w->state_count);
        arb_mat_set(w->rate_matrix, m->mat);
        _arb_mat_zero_diagonal(w->rate_matrix);
    }

    /*
     * Initialize the rate divisor, optionally using the equilibrium.
     * If the equilibrium is used, then the 'outer layer' rate divisor
     * will be ignored, and the 'middle layer' rate divisor will
     * be a function of the rate matrix and of the rate mixture expectation.
     */
    {
        w->rate_divisor = flint_malloc(sizeof(arb_struct));
        arb_init(w->rate_divisor);
        if (m->use_equilibrium_rate_divisor)
        {
            arb_struct *row_sums;
            row_sums = _arb_vec_init(w->state_count);
            _arb_mat_row_sums(row_sums, w->rate_matrix, prec);
            _arb_vec_dot(w->rate_divisor,
                    row_sums, w->equilibrium, w->state_count, w->prec);
            _arb_vec_clear(row_sums, w->state_count);
            if (m->rate_mixture->mode != RATE_MIXTURE_NONE)
            {
                arb_t tmp;
                arb_init(tmp);
                rate_mixture_expectation(tmp, m->rate_mixture, w->prec);
                /*
                if (!arb_is_one(tmp))
                {
                    flint_fprintf(stderr, "debug: expectation is not 1\n");
                    arb_fprintd(stderr, tmp, 15);
                }
                */
                arb_mul(w->rate_divisor, w->rate_divisor, tmp, w->prec);
                arb_clear(tmp);
            }
        }
        else
        {
            arb_set(w->rate_divisor, m->rate_divisor);
        }
    }

    /* fixme: remove */
    /* for backward compatibility, update m->rate_divisor */
    /* arb_set(m->rate_divisor, w->rate_divisor); */

    /*
     * Scale the rate matrix according to the rate divisor.
     * Each rate matrix associated with a specific rate category
     * and branch will need to be further scaled.
     */
    {
        arb_mat_scalar_div_arb(w->rate_matrix,
                w->rate_matrix, w->rate_divisor, w->prec);
    }

    /* Update the diagonal of the rate matrix. */
    _arb_update_rate_matrix_diagonal(w->rate_matrix, w->prec);

    /* allocate transition matrices */
    tmat_collection_init(w->transition_matrices,
        w->rate_category_count, w->edge_count, w->state_count);

    /* define transition probability matrices */
    {
        slong i, j;
        arb_t s;
        arb_init(s);
        for (i = 0; i < w->rate_category_count; i++)
        {
            for (j = 0; j < w->edge_count; j++)
            {
                arb_mat_struct *tmat;
                tmat = tmat_collection_entry(w->transition_matrices, i, j);
                rate_mixture_get_rate(s, m->rate_mixture, i);
                arb_mul(s, s, w->edge_rates + j, w->prec);
                arb_mat_scalar_mul_arb(tmat, w->rate_matrix, s, w->prec);
                arb_mat_exp(tmat, tmat, w->prec);
            }
        }
        arb_clear(s);
    }
}

void
cross_site_ws_clear(cross_site_ws_t w)
{
    if (w->edge_rates)
    {
        _arb_vec_clear(w->edge_rates, w->edge_count);
    }
    if (w->equilibrium)
    {
        _arb_vec_clear(w->equilibrium, w->state_count);
    }
    if (w->rate_divisor)
    {
        arb_clear(w->rate_divisor);
        flint_free(w->rate_divisor);
    }
    if (w->rate_matrix)
    {
        arb_mat_clear(w->rate_matrix);
        flint_free(w->rate_matrix);
    }
    tmat_collection_clear(w->transition_matrices);
}

void
cross_site_ws_reinit(cross_site_ws_t w, model_and_data_t m, slong prec)
{
    cross_site_ws_clear(w);
    cross_site_ws_init(w, m ,prec);
}
