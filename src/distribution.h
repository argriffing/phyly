#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H

#include "arb.h"
#include "arb_mat.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    slong n;
    arb_struct *p;
    int simplex_constraint;
} distribution_struct;
typedef distribution_struct distribution_t[1];

void distribution_init(distribution_t d, slong len);
void distribution_clear(distribution_t d);
arb_ptr distribution_entry(distribution_t d, slong i);
arb_srcptr distribution_srcentry(const distribution_t d, slong i);
void distribution_expectation_of_col(arb_t out,
        const distribution_struct *d, const arb_mat_t g,
        slong col, slong prec);
void distribution_set_col(distribution_t d, const arb_mat_t A, slong col,
        int simplex_constraint);
void distribution_set_vec(distribution_t d, const arb_struct *vec, slong len,
        int simplex_constraint);
int distribution_is_finite(const distribution_t d);

#ifdef __cplusplus
}
#endif

#endif
