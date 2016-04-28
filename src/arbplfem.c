/*
 * Full precision edge rate coefficient EM update.
 *
 * The JSON format is used for both input and output.
 * Arbitrary precision is used only internally;
 * double precision floating point without error bounds
 * is used for input and output.
 *
 * For now, site aggregation is required.
 * For now, edge reduction is forbidden.
 *
 * input format:
 * {
 * "model_and_data" : {
 *  "edges" : [[a, b], [c, d], ...],                 (#edges, 2)
 *  "edge_rate_coefficients" : [a, b, ...],          (#edges, )
 *  "rate_matrix" : [[a, b, ...], [c, d, ...], ...], (#states, #states)
 *  "probability_array" : [...]                      (#sites, #nodes, #states)
 * },
 * "site_reduction" : {
 *  "selection" : [a, b, c, ...], (optional)
 *  "aggregation" : {"sum" | "avg" | [a, b, c, ...]}
 * }}
 *
 * output format:
 * {
 *  "columns" : ["edge", "value"],
 *  "data" : [[a, b], [e, f], ..., [w, x]]
 * }
 *
 */
#include "jansson.h"
#include "flint/flint.h"

#include "arb_mat.h"
#include "arb.h"
#include "arf.h"

#include "csr_graph.h"
#include "model.h"
#include "reduction.h"
#include "util.h"
#include "evaluate_site_lhood.h"
#include "evaluate_site_marginal.h"
#include "arb_vec_extras.h"
#include "equilibrium.h"

#include "parsemodel.h"
#include "parsereduction.h"
#include "runjson.h"
#include "arbplfem.h"



/* Likelihood workspace. */
typedef struct
{
    int node_count;
    int edge_count;
    int state_count;
    arb_struct *edge_rates;
    arb_struct *equilibrium;
    arb_struct *dwell_accum;
    arb_struct *trans_accum;
    arb_mat_t rate_matrix;
    arb_mat_struct *transition_matrices;
    arb_mat_struct *dwell_frechet_matrices;
    arb_mat_struct *trans_frechet_matrices;
    arb_mat_struct *base_node_vectors;
    arb_mat_struct *lhood_node_vectors;
    arb_mat_struct *lhood_edge_vectors;
    arb_mat_struct *marginal_node_vectors;
} likelihood_ws_struct;
typedef likelihood_ws_struct likelihood_ws_t[1];

static void
likelihood_ws_init(likelihood_ws_t w, model_and_data_t m)
{
    csr_graph_struct *g;
    int i;
    arb_mat_struct *tmat;
    double tmpd;

    /*
     * This is the csr graph index of edge (a, b).
     * Given this index, node b is directly available
     * from the csr data structure.
     * The rate coefficient associated with the edge will also be available.
     * On the other hand, the index of node 'a' will be available through
     * the iteration order rather than directly from the index.
     */
    int idx;

    g = m->g;

    w->node_count = g->n;
    w->edge_count = g->nnz;
    w->state_count = arb_mat_nrows(m->mat);

    w->edge_rates = _arb_vec_init(w->edge_count);
    w->transition_matrices = flint_malloc(
            w->edge_count * sizeof(arb_mat_struct));
    w->trans_frechet_matrices = flint_malloc(
            w->edge_count * sizeof(arb_mat_struct));
    w->dwell_frechet_matrices = flint_malloc(
            w->edge_count * sizeof(arb_mat_struct));
    w->equilibrium = NULL;
    if (model_and_data_uses_equilibrium(m))
    {
        w->equilibrium = _arb_vec_init(w->state_count);
    }

    /* initialize the rate matrix */
    arb_mat_init(w->rate_matrix, w->state_count, w->state_count);

    /* intialize transition probability matrices */
    for (idx = 0; idx < w->edge_count; idx++)
    {
        tmat = w->transition_matrices + idx;
        arb_mat_init(tmat, w->state_count, w->state_count);
    }

    /* intialize frechet matrices */
    for (idx = 0; idx < w->edge_count; idx++)
    {
        tmat = w->trans_frechet_matrices + idx;
        arb_mat_init(tmat, w->state_count, w->state_count);

        tmat = w->dwell_frechet_matrices + idx;
        arb_mat_init(tmat, w->state_count, w->state_count);
    }

    /*
     * Define the map from csr edge index to edge rate.
     * The edge rate is represented in arbitrary precision,
     * but is assumed to take exactly the double precision input value.
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

    /* initialize per-node state vectors */
    w->base_node_vectors = flint_malloc(
            w->node_count * sizeof(arb_mat_struct));
    w->lhood_node_vectors = flint_malloc(
            w->node_count * sizeof(arb_mat_struct));
    w->marginal_node_vectors = flint_malloc(
            w->node_count * sizeof(arb_mat_struct));
    for (i = 0; i < w->node_count; i++)
    {
        arb_mat_init(w->base_node_vectors+i, w->state_count, 1);
        arb_mat_init(w->lhood_node_vectors+i, w->state_count, 1);
        arb_mat_init(w->marginal_node_vectors+i, w->state_count, 1);
    }

    /* initialize per-edge state vectors */
    w->lhood_edge_vectors = flint_malloc(
            w->edge_count * sizeof(arb_mat_struct));
    for (i = 0; i < w->edge_count; i++)
    {
        arb_mat_init(w->lhood_edge_vectors+i, w->state_count, 1);
    }

    /* init accumulators */
    w->dwell_accum = _arb_vec_init(w->edge_count);
    w->trans_accum = _arb_vec_init(w->edge_count);
}


static void
likelihood_ws_clear(likelihood_ws_t w)
{
    int i, idx;

    _arb_vec_clear(w->edge_rates, w->edge_count);
    if (w->equilibrium)
    {
        _arb_vec_clear(w->equilibrium, w->state_count);
    }

    /* clear unscaled rate matrix */
    arb_mat_clear(w->rate_matrix);

    /* clear per-edge matrices */
    for (idx = 0; idx < w->edge_count; idx++)
    {
        arb_mat_clear(w->transition_matrices + idx);
        arb_mat_clear(w->dwell_frechet_matrices + idx);
        arb_mat_clear(w->trans_frechet_matrices + idx);
        arb_mat_clear(w->lhood_edge_vectors + idx);
    }
    flint_free(w->transition_matrices);
    flint_free(w->dwell_frechet_matrices);
    flint_free(w->trans_frechet_matrices);
    flint_free(w->lhood_edge_vectors);

    /* clear per-node matrices */
    for (i = 0; i < w->node_count; i++)
    {
        arb_mat_clear(w->base_node_vectors + i);
        arb_mat_clear(w->lhood_node_vectors + i);
        arb_mat_clear(w->marginal_node_vectors + i);
    }
    flint_free(w->base_node_vectors);
    flint_free(w->lhood_node_vectors);
    flint_free(w->marginal_node_vectors);

    /* clear accumulators */
    _arb_vec_clear(w->dwell_accum, w->edge_count);
    _arb_vec_clear(w->trans_accum, w->edge_count);
}

static void
likelihood_ws_update(likelihood_ws_t w, model_and_data_t m,
        const int *edge_is_requested, slong prec)
{
    /* arrays are already allocated and initialized */
    int idx;
    arb_mat_t P, L;
    arb_struct *rates_out;

    _update_rate_matrix_and_equilibrium(
            w->rate_matrix,
            w->equilibrium,
            m->rate_divisor,
            m->use_equilibrium_rate_divisor,
            m->root_prior,
            m->mat,
            prec);

    /* clear accumulators */
    _arb_vec_zero(w->dwell_accum, w->edge_count);
    _arb_vec_zero(w->trans_accum, w->edge_count);

    /* utility matrices */
    arb_mat_init(P, w->state_count, w->state_count);
    arb_mat_init(L, w->state_count, w->state_count);

    /* compute rates out */
    rates_out = _arb_vec_init(w->state_count);
    _arb_mat_row_sums(rates_out, w->rate_matrix, prec);

    /*
     * Update state->state matrices on each edge.
     * The transition matrix will be updated on each edge.
     * The frechet exp matrices will be updated only on requested edges.
     */
    for (idx = 0; idx < w->edge_count; idx++)
    {
        int state;
        arb_mat_struct *tmat;
        tmat = w->transition_matrices + idx;
        arb_mat_set(tmat, w->rate_matrix);
        for (state = 0; state < w->state_count; state++)
        {
            arb_sub(
                    arb_mat_entry(tmat, state, state),
                    arb_mat_entry(tmat, state, state),
                    rates_out + state, prec);
        }
        arb_mat_scalar_mul_arb(tmat, tmat, w->edge_rates + idx, prec);

        /* at this point, tmat is the scaled rate matrix with zero row sums */

        /* update the dwell frechet matrix for the current edge */
        /* set diagonal entries of L to the rates out of that state */
        if (edge_is_requested[idx])
        {
            arb_mat_struct *fmat;
            fmat = w->dwell_frechet_matrices + idx;

            /* set diagonals of L to rate-out */
            arb_mat_zero(L);
            for (state = 0; state < w->state_count; state++)
                arb_set(arb_mat_entry(L, state, state), rates_out + state);

            _arb_mat_exp_frechet(P, fmat, tmat, L, prec);
        }

        /* update the trans frechet matrix for the current edge */
        if (edge_is_requested[idx])
        {
            arb_mat_struct *fmat;
            fmat = w->trans_frechet_matrices + idx;

            /*
             * Note that w->rate matrix has zeros on the diagonal
             * and it is not scaled by the edge rate.
             */
            _arb_mat_exp_frechet(P, fmat, tmat, w->rate_matrix, prec);
        }

        /* compute the exponential of the rate matrix */
        arb_mat_exp(tmat, tmat, prec);
    }

    arb_mat_clear(P);
    arb_mat_clear(L);
    _arb_vec_clear(rates_out, w->state_count);
}


static void
evaluate_edge_expectations(
        arb_struct *dwell_accum,
        arb_struct *trans_accum,
        arb_mat_struct *marginal_node_vectors,
        arb_mat_struct *lhood_node_vectors,
        arb_mat_struct *lhood_edge_vectors,
        arb_mat_struct *dwell_frechet_matrices,
        arb_mat_struct *trans_frechet_matrices,
        csr_graph_t g, int *preorder,
        int node_count, int state_count,
        const int *edge_is_requested, slong prec)
{
    int u, a, b;
    int idx;
    int start, stop;
    slong state;
    arb_mat_struct *lvec, *mvec, *evec;
    arb_mat_t fvec;
    arb_t tmp, dwell_tmp, trans_tmp;

    arb_mat_init(fvec, state_count, 1);

    arb_init(tmp);
    arb_init(dwell_tmp);
    arb_init(trans_tmp);

    for (u = 0; u < node_count; u++)
    {
        a = preorder[u];
        mvec = marginal_node_vectors + a;
        start = g->indptr[a];
        stop = g->indptr[a+1];

        for (idx = start; idx < stop; idx++)
        {
            /* flint_printf("debug: idx=%d\n", idx); */
            b = g->indices[idx];
            /*
             * At this point (a, b) is an edge from node a to node b
             * in a pre-order traversal of edges of the tree.
             */
            lvec = lhood_node_vectors + b;
            evec = lhood_edge_vectors + idx;

            /*
            flint_printf("debug: edge idx %d is requested? %d\n",
                    idx, edge_is_requested[idx]);
            */

            /* dwell update */
            if (edge_is_requested[idx])
            {
                arb_zero(dwell_tmp);
                arb_mat_mul(fvec, dwell_frechet_matrices + idx, lvec, prec);
                for (state = 0; state < state_count; state++)
                {
                    if (!arb_is_zero(arb_mat_entry(evec, state, 0)))
                    {
                        arb_div(tmp,
                                arb_mat_entry(fvec, state, 0),
                                arb_mat_entry(evec, state, 0), prec);
                        arb_addmul(
                                dwell_tmp,
                                arb_mat_entry(mvec, state, 0),
                                tmp, prec);
                    }
                }

                /*
                flint_printf("debug: dwell_tmp = ");
                arb_printd(dwell_tmp, 15); flint_printf("\n");
                */

                arb_set(dwell_accum + idx, dwell_tmp);
            }

            /* trans update */
            if (edge_is_requested[idx])
            {
                arb_zero(trans_tmp);
                arb_mat_mul(fvec, trans_frechet_matrices + idx, lvec, prec);
                for (state = 0; state < state_count; state++)
                {
                    if (!arb_is_zero(arb_mat_entry(evec, state, 0)))
                    {
                        arb_div(tmp,
                                arb_mat_entry(fvec, state, 0),
                                arb_mat_entry(evec, state, 0), prec);
                        arb_addmul(
                                trans_tmp,
                                arb_mat_entry(mvec, state, 0),
                                tmp, prec);
                    }
                }
                /*
                flint_printf("debug: trans_tmp = ");
                arb_printd(trans_tmp, 15); flint_printf("\n");
                */
                arb_set(trans_accum + idx, trans_tmp);
            }
        }
    }

    arb_clear(tmp);
    arb_clear(dwell_tmp);
    arb_clear(trans_tmp);
    arb_mat_clear(fvec); 
}


static int
_accum(likelihood_ws_t w, model_and_data_t m, column_reduction_t r_site, 
        const int *edge_is_requested, slong prec)
{
    int site, idx, i;
    arb_t lhood;
    int site_count;
    int result;
    arb_struct *dwell_site, *trans_site;

    arb_t site_weight_divisor;
    arb_struct *site_weights;
    int *site_selection_count;

    arb_init(lhood);
    result = 0;
    site_count = pmat_nsites(m->p);

    dwell_site = _arb_vec_init(w->edge_count);
    trans_site = _arb_vec_init(w->edge_count);

    arb_init(site_weight_divisor);
    site_weights = _arb_vec_init(site_count);
    site_selection_count = calloc(site_count, sizeof(int));

    /* count how many times each site index is included in the selection */
    for (i = 0; i < r_site->selection_len; i++)
    {
        site = r_site->selection[i];
        site_selection_count[site]++;
    }

    /* define site aggregation weights */
    result = get_column_agg_weights(
            site_weight_divisor, site_weights, site_count, r_site, prec);
    if (result) goto finish;

    for (site = 0; site < site_count; site++)
    {
        if (!site_selection_count[site])
            continue;

        /* update base node vectors */
        pmat_update_base_node_vectors(
                w->base_node_vectors, m->p, site,
                m->use_equilibrium_root_prior, w->equilibrium,
                m->preorder[0], prec);

        /*
         * Update per-node and per-edge likelihood vectors.
         * Actually the likelihood vectors on edges are not used.
         * This is a backward pass from the leaves to the root.
         */
        evaluate_site_lhood(lhood,
                w->lhood_node_vectors,
                w->lhood_edge_vectors,
                w->base_node_vectors,
                w->transition_matrices,
                m->g, m->preorder, w->node_count, prec);

        /*
         * Update marginal distribution vectors at nodes.
         * This is a forward pass from the root to the leaves.
         */
        evaluate_site_marginal(
                w->marginal_node_vectors,
                w->lhood_node_vectors,
                w->lhood_edge_vectors,
                w->transition_matrices,
                m->g, m->preorder, w->node_count, w->state_count, prec);

        /* Update dwell and trans expectations on edges. */
        evaluate_edge_expectations(
                dwell_site,
                trans_site,
                w->marginal_node_vectors,
                w->lhood_node_vectors,
                w->lhood_edge_vectors,
                w->dwell_frechet_matrices,
                w->trans_frechet_matrices,
                m->g, m->preorder, w->node_count, w->state_count,
                edge_is_requested, prec);

        /* Accumulate, using the site weights and weight divisor. */
        {
            arb_t tmp;
            arb_init(tmp);
            arb_div(tmp, site_weights + site, site_weight_divisor, prec);
            for (idx = 0; idx < w->edge_count; idx++)
            {
                if (!edge_is_requested[idx])
                    continue;

                arb_addmul(w->dwell_accum + idx, dwell_site + idx, tmp, prec);
                arb_addmul(w->trans_accum + idx, trans_site + idx, tmp, prec);
            }
            arb_clear(tmp);
        }
    }

finish:

    arb_clear(lhood);
    arb_clear(site_weight_divisor);
    _arb_vec_clear(site_weights, site_count);
    _arb_vec_clear(dwell_site, w->edge_count);
    _arb_vec_clear(trans_site, w->edge_count);
    free(site_selection_count);

    return result;
}


static json_t *
_query(model_and_data_t m,
        column_reduction_t r_site,
        int *result_out)
{
    json_t * j_out = NULL;
    slong prec;
    slong idx;
    likelihood_ws_t w;
    arb_struct *final;
    int result, success;
    int *edge_is_requested;

    result = 0;

    /* initialize likelihood workspace */
    likelihood_ws_init(w, m);

    edge_is_requested = flint_malloc(w->edge_count * sizeof(int));
    for (idx = 0; idx < w->edge_count; idx++)
        edge_is_requested[idx] = 1;

    /* initialize output vector */
    final = _arb_vec_init(w->edge_count);

    /* repeat with increasing precision until there is no precision failure */
    for (success = 0, prec=4; !success; prec <<= 1)
    {
        /* this does not update per-edge or per-node likelihood vectors */
        likelihood_ws_update(w, m, edge_is_requested, prec);

        /* accumulate numerators and denominators over sites */
        _accum(w, m, r_site, edge_is_requested, prec);

        /*
         * For each edge, compute the ratio of two values that have been
         * accumulated over sites.
         * If the conditionally expected number of transitions is exactly
         * zero, then set the ratio to zero regardless of the
         * expected rates out of the occupied states, even if those
         * rates are zero.
         */
        for (idx = 0; idx < w->edge_count; idx++)
        {
            if (!edge_is_requested[idx])
                continue;

            /*
            flint_printf("debug (prec=%wd idx=%wd): "
                         "trans accum:\n", prec, idx);
            arb_printd(w->trans_accum + idx, 15); flint_printf("\n");

            flint_printf("debug (prec=%wd idx=%wd): "
                         "dwell accum:\n", prec, idx);
            arb_printd(w->dwell_accum + idx, 15); flint_printf("\n");
            */

            if (arb_is_zero(w->trans_accum + idx))
            {
                arb_zero(final + idx);
            }
            else
            {
                arb_div(final + idx,
                        w->trans_accum + idx, w->dwell_accum + idx, prec);
            }
            arb_mul(final + idx, final + idx, w->edge_rates + idx, prec);
        }

        /*
        flint_printf("debug:\n");
        flint_printf("prec=%wd\n", prec);

        flint_printf("numerator(trans)=\n");
        _arb_vec_print(w->trans_accum, w->edge_count);
        flint_printf("\n");

        flint_printf("denominator(dwell)=\n");
        _arb_vec_print(w->dwell_accum, w->edge_count);
        flint_printf("\n");

        flint_printf("ratio=\n");
        _arb_vec_print(final, w->edge_count);
        flint_printf("\n");
        */


        /* check which entries are accurate to full relative precision  */
        success = 1;
        for (idx = 0; idx < w->edge_count; idx++)
        {
            if (_can_round(final + idx))
            {
                edge_is_requested[idx] = 0;

                /*
                flint_printf("debug: edge %wd is now accurate to full "
                             "relative precision with value \n");
                arb_printd(final + idx, 15);
                flint_printf("\n");
                */
            }
            else
            {
                success = 0;
            }
        }
    }

    /* build the json output */
    {
        double d;
        int edge, idx;
        json_t *j_data, *x;
        j_data = json_array();
        for (edge = 0; edge < w->edge_count; edge++)
        {
            idx = m->edge_map->order[edge];
            d = arf_get_d(arb_midref(final + idx), ARF_RND_NEAR);
            x = json_pack("[i, f]", edge, d);
            json_array_append_new(j_data, x);
        }
        j_out = json_pack("{s:[s, s], s:o}",
                "columns", "edge", "value",
                "data", j_data);
    }

    /* clear output vector */
    _arb_vec_clear(final, w->edge_count);

    /* clear likelihood workspace */
    likelihood_ws_clear(w);

    flint_free(edge_is_requested);

    *result_out = result;
    return j_out;
}


static int
_parse(model_and_data_t m, column_reduction_t r_site, json_t *root)
{
    json_t *model_and_data = NULL;
    json_t *site_reduction = NULL;
    int result;

    result = 0;

    /* unpack the top level of json input */
    {
        size_t flags;
        json_error_t err;
        flags = JSON_STRICT;
        result = json_unpack_ex(root, &err, flags,
                "{s:o, s?o}",
                "model_and_data", &model_and_data,
                "site_reduction", &site_reduction
                );
        if (result)
        {
            fprintf(stderr, "error: on line %d: %s\n", err.line, err.text);
            return result;
        }
    }

    /* validate the model and data section of the json input */
    result = validate_model_and_data(m, model_and_data);
    if (result) return result;

    /* validate the site reduction section of the json input */
    {
        int site_count;
        site_count = pmat_nsites(m->p);
        result = validate_column_reduction(
                r_site, site_count, "site", site_reduction);
        if (result) return result;
    }

    return result;
}


json_t *arbplf_em_update_run(void *userdata, json_t *root, int *retcode)
{
    json_t *j_out = NULL;
    model_and_data_t m;
    column_reduction_t r_site;
    int result = 0;

    model_and_data_init(m);
    column_reduction_init(r_site);

    if (userdata)
    {
        fprintf(stderr, "internal error: unexpected userdata\n");
        result = -1;
        goto finish;
    }

    result = _parse(m, r_site, root);
    if (result) goto finish;

    if (r_site->agg_mode == AGG_NONE)
    {
        fprintf(stderr, "error: aggregation over sites is required\n");
        result = -1;
        goto finish;
    }

    j_out = _query(m, r_site, &result);
    if (result) goto finish;

finish:

    *retcode = result;

    model_and_data_clear(m);
    column_reduction_clear(r_site);

    flint_cleanup();
    return j_out;
}
