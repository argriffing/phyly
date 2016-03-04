#ifndef ARBPLFCOEFFEXPECT_H
#define ARBPLFCOEFFEXPECT_H

#include "jansson.h"


#ifdef __cplusplus
extern "C" {
#endif

json_t *arbplf_coeff_expect_run(void *userdata, json_t *root, int *retcode);

#ifdef __cplusplus
}
#endif

#endif
