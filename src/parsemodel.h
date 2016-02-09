#ifndef PARSEMODEL_H
#define PARSEMODEL_H

#include "jansson.h"

#include "model.h"


#ifdef __cplusplus
extern "C" {
#endif

/*
 * On success returns 0.
 * On failure, writes a message to stderr and returns nonzero.
 */
int validate_model_and_data(model_and_data_t m, json_t *root);


#ifdef __cplusplus
}
#endif

#endif
