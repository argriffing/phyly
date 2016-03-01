#ifndef ARBPLFDWELL_H
#define ARBPLFDWELL_H

#include "jansson.h"


#ifdef __cplusplus
extern "C" {
#endif

json_t *arbplf_dwell_run(void *userdata, json_t *root, int *retcode);

#ifdef __cplusplus
}
#endif

#endif
