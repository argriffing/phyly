#ifndef PARSEREDUCTION_H
#define PARSEREDUCTION_H

#include "jansson.h"

#include "reduction.h"


#ifdef __cplusplus
extern "C" {
#endif

int validate_column_reduction(column_reduction_t r,
        int k, const char *name, json_t *root);

#ifdef __cplusplus
}
#endif

#endif
