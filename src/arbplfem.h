#ifndef ARBPLFCOEFFEXPECT_H
#define ARBPLFCOEFFEXPECT_H

#include "jansson.h"


#ifdef __cplusplus
extern "C" {
#endif

json_t *arbplf_em_update_run(void *userdata, json_t *root, int *retcode);

#ifdef __cplusplus
}
#endif

#endif
