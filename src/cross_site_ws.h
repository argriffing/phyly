#ifndef CROSS_SITE_WS_H
#define CROSS_SITE_WS_H

#include "flint/flint.h"
#include "arb.h"
#include "arb_mat.h"

#ifdef __cplusplus
extern "C" {
#endif



/* this is just a two dimensional array of arb matrices */
typedef struct
{
    slong rate_category_count;
    slong edge_count;
    arb_mat_struct *matrices;
} tmat_collection_struct;
typedef tmat_collection_struct tmat_collection_t[1];

void tmat_collection_pre_init(tmat_collection_t x);
void tmat_collection_init(tmat_collection_t x,
        slong rate_category_count, slong edge_count, slong state_count);
void tmat_collection_clear(tmat_collection_t x);
arb_mat_struct * tmat_collection_entry(tmat_collection_t x,
        slong rate_category_idx, slong edge_idx);



typedef struct
{
    slong prec;
    slong rate_category_count;
    slong node_count;
    slong edge_count;
    slong state_count;
    arb_struct *edge_rates;
    arb_struct *equilibrium;
    arb_struct *rate_divisor;
    arb_mat_struct *rate_matrix;
    tmat_collection_t transition_matrices;
} cross_site_ws_struct;
typedef cross_site_ws_struct cross_site_ws_t[1];

void cross_site_ws_pre_init(cross_site_ws_t w);
void cross_site_ws_init(cross_site_ws_t w, model_and_data_t m, slong prec);
void cross_site_ws_clear(cross_site_ws_t w);
void cross_site_ws_reinit(cross_site_ws_t w, model_and_data_t m, slong prec);


#ifdef __cplusplus
}
#endif

#endif
