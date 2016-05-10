#ifndef CROSS_SITE_WS_H
#define CROSS_SITE_WS_H

#include "flint/flint.h"
#include "arb.h"
#include "arb_mat.h"

#ifdef __cplusplus
extern "C" {
#endif

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

    /* matrices for each category and edge */
    arb_mat_struct *transition_matrices;
    arb_mat_struct *dwell_frechet_matrices;
    arb_mat_struct *trans_frechet_matrices;

} cross_site_ws_struct;
typedef cross_site_ws_struct cross_site_ws_t[1];

void cross_site_ws_pre_init(cross_site_ws_t w);
void cross_site_ws_init(cross_site_ws_t w, const model_and_data_t m);
void cross_site_ws_init_dwell(cross_site_ws_t w);
void cross_site_ws_init_trans(cross_site_ws_t w);
void cross_site_ws_update(cross_site_ws_t w, model_and_data_t m, slong prec);
void cross_site_ws_clear(cross_site_ws_t w);

arb_mat_struct * cross_site_ws_transition_matrix(cross_site_ws_t w,
        slong rate_category_idx, slong edge_idx);

arb_mat_struct * cross_site_ws_dwell_frechet_matrix(cross_site_ws_t w,
        slong rate_category_idx, slong edge_idx);

arb_mat_struct * cross_site_ws_trans_frechet_matrix(cross_site_ws_t w,
        slong rate_category_idx, slong edge_idx);

#ifdef __cplusplus
}
#endif

#endif
