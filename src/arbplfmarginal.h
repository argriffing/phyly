#ifndef ARBPLFMARGINAL_H
#define ARBPLFMARGINAL_H

#include "jansson.h"


#ifdef __cplusplus
extern "C" {
#endif

json_t *arbplf_marginal_run(void *userdata, json_t *root, int *retcode);

#ifdef __cplusplus
}
#endif

#endif
