#ifndef ARBPLFDERIV_H
#define ARBPLFDERIV_H

#include "jansson.h"

#ifdef __cplusplus
extern "C" {
#endif

json_t *arbplf_deriv_run(void *userdata, json_t *root, int *retcode);

#ifdef __cplusplus
}
#endif

#endif
