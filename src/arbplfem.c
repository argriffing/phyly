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
#include "arb_mat_extras.h"
#include "equilibrium.h"
#include "cross_site_ws.h"

#include "parsemodel.h"
#include "parsereduction.h"
#include "runjson.h"
#include "arbplfem.h"


typedef struct
{
    arb_struct *dwell_accum;
    arb_struct *trans_accum;
    arb_mat_struct *base_node_vectors;
    arb_mat_struct *lhood_node_vectors;
    arb_mat_struct *lhood_edge_vectors;
    arb_mat_struct *marginal_node_vectors;
} likelihood_ws_struct;
typedef likelihood_ws_struct likelihood_ws_t[1];

static void
likelihood_ws_init(likelihood_ws_t w, const model_and_data_t m)
{
    slong node_count = model_and_data_node_count(m);
    slong edge_count = model_and_data_edge_count(m);
    slong state_count = model_and_data_state_count(m);

    w->dwell_accum = _arb_vec_init(edge_count);
    w->trans_accum = _arb_vec_init(edge_count);
    w->lhood_edge_vectors = _arb_mat_vec_init(state_count, 1, edge_count);
    w->base_node_vectors = _arb_mat_vec_init(state_count, 1, node_count);
    w->lhood_node_vectors = _arb_mat_vec_init(state_count, 1, node_count);
    w->marginal_node_vectors = _arb_mat_vec_init(state_count, 1, node_count);
}

static void
likelihood_ws_clear(likelihood_ws_t w, const model_and_data_t m)
{
    slong node_count = model_and_data_node_count(m);
    slong edge_count = model_and_data_edge_count(m);

    _arb_vec_clear(w->dwell_accum, edge_count);
    _arb_vec_clear(w->trans_accum, edge_count);
    _arb_mat_vec_clear(w->lhood_edge_vectors, edge_count);
    _arb_mat_vec_clear(w->base_node_vectors, node_count);
    _arb_mat_vec_clear(w->lhood_node_vectors, node_count);
    _arb_mat_vec_clear(w->marginal_node_vectors, node_count);
}

static void
_update_frechet_matrices(cross_site_ws_t csw, model_and_data_t m, slong prec)
{
    slong state, cat, idx, sa, sb;
    arb_mat_t P, L_dwell, L_trans, Q;
    arb_t cat_rate, rate;

    slong state_count = model_and_data_state_count(m);
    slong edge_count = model_and_data_edge_count(m);
    slong rate_category_count = model_and_data_rate_category_count(m);

    arb_init(cat_rate);
    arb_init(rate);
    arb_mat_init(P, state_count, state_count);
    arb_mat_init(L_dwell, state_count, state_count);
    arb_mat_init(L_trans, state_count, state_count);
    arb_mat_init(Q, state_count, state_count);

    /* L_dwell has exit rates on diagonals and zeros on off-diagonals */
    for (state = 0; state < state_count; state++)
        arb_neg(arb_mat_entry(L_dwell, state, state),
                arb_mat_entry(csw->rate_matrix, state, state));

    /* L_trans has zeros on diagonals and rates on off-diagonals */
    for (sa = 0; sa < state_count; sa++)
        for (sb = 0; sb < state_count; sb++)
            if (sa != sb)
                arb_set(arb_mat_entry(L_trans, sa, sb),
                        arb_mat_entry(csw->rate_matrix, sa, sb));

    for (cat = 0; cat < rate_category_count; cat++)
    {
        rate_mixture_get_rate(cat_rate, m->rate_mixture, cat);
        for (idx = 0; idx < edge_count; idx++)
        {
            arb_mat_struct *fmat;

            /* scale the rate matrix */
            arb_mul(rate, csw->edge_rates + idx, cat_rate, prec);
            arb_mat_scalar_mul_arb(Q, csw->rate_matrix, rate, prec);

            /* update dwell */
            fmat = cross_site_ws_dwell_frechet_matrix(csw, cat, idx);
            _arb_mat_exp_frechet(P, fmat, Q, L_dwell, prec);

            /* update trans */
            fmat = cross_site_ws_trans_frechet_matrix(csw, cat, idx);
            _arb_mat_exp_frechet(P, fmat, Q, L_trans, prec);
        }
    }

    arb_clear(cat_rate);
    arb_clear(rate);
    arb_mat_clear(P);
    arb_mat_clear(L_dwell);
    arb_mat_clear(L_trans);
    arb_mat_clear(Q);
}

static void
_evaluate_edge_expectations(
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
            b = g->indices[idx];

            /*
             * At this point (a, b) is an edge from node a to node b
             * in a pre-order traversal of edges of the tree.
             */
            lvec = lhood_node_vectors + b;
            evec = lhood_edge_vectors + idx;

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
_accum(likelihood_ws_t w, cross_site_ws_t csw, model_and_data_t m,
        column_reduction_t r_site, const int *edge_is_requested, slong prec)
{
    slong cat, site, idx, i;
    arb_t tmp, cat_rate;
    arb_t site_lhood, cat_lhood, prior_prob, lhood;
    arb_struct *dwell_site, *trans_site;
    arb_struct *dwell_cat, *trans_cat;
    int result = 0;

    arb_t site_weight_divisor;
    arb_struct *site_weights;
    int *site_selection_count;

    slong edge_count = model_and_data_edge_count(m);
    slong site_count = model_and_data_site_count(m);
    slong node_count = model_and_data_node_count(m);
    slong state_count = model_and_data_state_count(m);
    slong rate_category_count = model_and_data_rate_category_count(m);

    arb_init(tmp);
    arb_init(cat_rate);

    arb_init(site_lhood);
    arb_init(cat_lhood);
    arb_init(prior_prob);
    arb_init(lhood);

    dwell_site = _arb_vec_init(edge_count);
    trans_site = _arb_vec_init(edge_count);
    dwell_cat = _arb_vec_init(edge_count);
    trans_cat = _arb_vec_init(edge_count);

    arb_init(site_weight_divisor);
    site_weights = _arb_vec_init(site_count);
    site_selection_count = calloc(site_count, sizeof(int));

    /* clear accumulators */
    _arb_vec_zero(w->dwell_accum, edge_count);
    _arb_vec_zero(w->trans_accum, edge_count);

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
                m->root_prior, csw->equilibrium,
                m->preorder[0], prec);

        /* clear cross-category expectations and site lhood */
        _arb_vec_zero(dwell_site, edge_count);
        _arb_vec_zero(trans_site, edge_count);
        arb_zero(site_lhood);

        for (cat = 0; cat < rate_category_count; cat++)
        {
            arb_mat_struct *tmat_base, *dwell_base, *trans_base;
            tmat_base = cross_site_ws_transition_matrix(csw, cat, 0);
            dwell_base = cross_site_ws_dwell_frechet_matrix(csw, cat, 0);
            trans_base = cross_site_ws_trans_frechet_matrix(csw, cat, 0);

            /*
             * Update per-node and per-edge likelihood vectors.
             * Actually the likelihood vectors on edges are not used.
             * This is a backward pass from the leaves to the root.
             */
            evaluate_site_lhood(lhood,
                    w->lhood_node_vectors,
                    w->lhood_edge_vectors,
                    w->base_node_vectors,
                    tmat_base,
                    m->g, m->preorder, node_count, prec);

            /* Compute the likelihood for the rate category. */
            rate_mixture_get_prob(prior_prob, m->rate_mixture, cat, prec);
            arb_mul(cat_lhood, lhood, prior_prob, prec);
            arb_add(site_lhood, site_lhood, cat_lhood, prec);

            if (!arb_is_zero(cat_lhood))
            {
                rate_mixture_get_rate(cat_rate, m->rate_mixture, cat);

                /*
                 * Update marginal distribution vectors at nodes.
                 * This is a forward pass from the root to the leaves.
                 */
                evaluate_site_marginal(
                        w->marginal_node_vectors,
                        w->lhood_node_vectors,
                        w->lhood_edge_vectors,
                        tmat_base,
                        m->g, m->preorder, node_count, state_count, prec);

                /* Update dwell and trans expectations on edges. */
                _evaluate_edge_expectations(
                        dwell_cat,
                        trans_cat,
                        w->marginal_node_vectors,
                        w->lhood_node_vectors,
                        w->lhood_edge_vectors,
                        dwell_base,
                        trans_base,
                        m->g, m->preorder, node_count, state_count,
                        edge_is_requested, prec);

                /* Accumulate the category-specific expectations. */
                for (idx = 0; idx < edge_count; idx++)
                {
                    if (!edge_is_requested[idx]) continue;
                    arb_mul(tmp, cat_lhood, cat_rate, prec);
                    arb_addmul(dwell_site + idx, dwell_cat + idx, tmp, prec);
                    arb_addmul(trans_site + idx, trans_cat + idx, tmp, prec);
                }
            }
        }

        /* Accumulate expectations across sites. */
        {
            arb_div(tmp, site_weights + site, site_weight_divisor, prec);
            arb_div(tmp, tmp, site_lhood, prec);
            for (idx = 0; idx < edge_count; idx++)
            {
                if (!edge_is_requested[idx]) continue;
                arb_addmul(w->dwell_accum + idx, dwell_site + idx, tmp, prec);
                arb_addmul(w->trans_accum + idx, trans_site + idx, tmp, prec);
            }
        }
    }

finish:

    arb_clear(tmp);
    arb_clear(cat_rate);

    arb_clear(site_lhood);
    arb_clear(cat_lhood);
    arb_clear(prior_prob);
    arb_clear(lhood);

    arb_clear(site_weight_divisor);
    _arb_vec_clear(site_weights, site_count);
    _arb_vec_clear(dwell_site, edge_count);
    _arb_vec_clear(trans_site, edge_count);
    _arb_vec_clear(dwell_cat, edge_count);
    _arb_vec_clear(trans_cat, edge_count);
    free(site_selection_count);

    return result;
}


static json_t *
_query(model_and_data_t m, column_reduction_t r_site, int *result_out)
{
    json_t * j_out = NULL;
    slong prec;
    slong idx;
    cross_site_ws_t csw;
    likelihood_ws_t w;
    arb_struct *final;
    int success;
    int *edge_is_requested;
    int result = 0;

    slong edge_count = model_and_data_edge_count(m);

    /* initialize likelihood workspace */
    likelihood_ws_init(w, m);
    cross_site_ws_init(csw, m);
    cross_site_ws_init_dwell(csw);
    cross_site_ws_init_trans(csw);

    edge_is_requested = flint_malloc(edge_count * sizeof(int));
    for (idx = 0; idx < edge_count; idx++)
        edge_is_requested[idx] = 1;

    /* initialize output vector */
    final = _arb_vec_init(edge_count);

    /* repeat with increasing precision until there is no precision failure */
    for (success = 0, prec=4; !success; prec <<= 1)
    {
        /* update transition rate and transition probability matrices */
        cross_site_ws_update(csw, m, prec);

        /* update frechet dwell and trans matrices */
        _update_frechet_matrices(csw, m, prec);

        /* accumulate numerators and denominators over sites */
        _accum(w, csw, m, r_site, edge_is_requested, prec);

        /*
         * For each edge, compute the ratio of two values that have been
         * accumulated over sites.
         * If the conditionally expected number of transitions is exactly
         * zero, then set the ratio to zero regardless of the
         * expected rates out of the occupied states, even if those
         * rates are zero.
         */
        for (idx = 0; idx < edge_count; idx++)
        {
            if (!edge_is_requested[idx])
                continue;

            if (arb_is_zero(w->trans_accum + idx))
            {
                arb_zero(final + idx);
            }
            else
            {
                arb_div(final + idx,
                        w->trans_accum + idx, w->dwell_accum + idx, prec);
            }
            arb_mul(final + idx, final + idx, csw->edge_rates + idx, prec);
        }

        /* check which entries are accurate to full relative precision  */
        success = 1;
        for (idx = 0; idx < edge_count; idx++)
        {
            if (_can_round(final + idx))
            {
                edge_is_requested[idx] = 0;
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
        for (edge = 0; edge < edge_count; edge++)
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
    _arb_vec_clear(final, edge_count);

    /* clear likelihood workspace */
    cross_site_ws_clear(csw);
    likelihood_ws_clear(w, m);

    flint_free(edge_is_requested);

    *result_out = result;
    return j_out;
}


static int
_parse(model_and_data_t m, column_reduction_t r_site, json_t *root)
{
    json_t *model_and_data = NULL;
    json_t *site_reduction = NULL;
    int result = 0;

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
        slong site_count = model_and_data_site_count(m);

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
