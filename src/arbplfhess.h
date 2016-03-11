#ifndef ARBPLFHESS_H
#define ARBPLFHESS_H

#include "jansson.h"

#include "model.h"
#include "reduction.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef json_t * (* second_order_query_t)(
        model_and_data_t m, column_reduction_t r_site, int *result_out);

json_t *arbplf_second_order_run(void *userdata, json_t *root, int *retcode);


json_t * newton_point_query(
        model_and_data_t m, column_reduction_t r_site, int *result_out);

json_t * newton_delta_query(
        model_and_data_t m, column_reduction_t r_site, int *result_out);

json_t * hess_query(
        model_and_data_t m, column_reduction_t r_site, int *result_out);

json_t * inv_hess_query(
        model_and_data_t m, column_reduction_t r_site, int *result_out);

json_t * opt_cert_query(
        model_and_data_t m, column_reduction_t r_site, int *result_out);


#ifdef __cplusplus
}
#endif

#endif
