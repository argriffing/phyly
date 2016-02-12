#ifndef ARBPLFHESS_H
#define ARBPLFHESS_H

#include "jansson.h"

#ifdef __cplusplus
extern "C" {
#endif

json_t *arbplf_hess_run(void *userdata, json_t *root, int *retcode);

#ifdef __cplusplus
}
#endif

#endif
