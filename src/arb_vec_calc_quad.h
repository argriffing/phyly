#ifndef ARB_VEC_CALC_QUAD_H
#define ARB_VEC_CALC_QUAD_H

#include "flint/flint.h"

#include "arf.h"
#include "arb.h"
#include "arb_mat.h"

#include "arb_vec_extras.h"


#ifdef __cplusplus
extern "C" {
#endif

/* extern TLS_PREFIX int arb_vec_calc_verbose; */


/* multivariate function (R^N -> R) with gradient and hessian */
typedef int (*arb_vec_calc_f_t) (
        arb_struct *f, arb_struct *g, arb_mat_struct *h,
        const arb_struct *x, void *param, slong n, slong prec);

/* quadratic local approximation of a function (f : R^N -> R) */
typedef struct
{
    slong n;
    arb_struct *x;
    arb_struct *y;
    arb_struct *g;
    arb_mat_struct *h;
    arb_struct *p_cauchy;
    arb_struct *p_newton;
    arb_vec_calc_f_t f;
    void *param;
    slong prec;
} quad_struct;
typedef quad_struct myquad_t[1];


void
quad_init(myquad_t q, arb_vec_calc_f_t f,
        arb_struct *x, void *param, slong n, slong prec);

void quad_clear(myquad_t q);
void quad_init_set(myquad_t b, const myquad_t a);
void quad_set(myquad_t b, const myquad_t a);
void quad_printd(const myquad_t q, slong digits);

arb_struct * quad_alloc_y(myquad_t q);
arb_struct * quad_alloc_g(myquad_t q);
arb_mat_struct * quad_alloc_h(myquad_t q);

/* todo: move this to an optimization-specific file? */
void _minimize_dogleg(myquad_t q_opt, myquad_t q_initial,
        const arb_t initial_radius, const arb_t max_radius, slong maxiter);


/* todo: move this to an optimization-specific file? */
void _solve_dogleg_subproblem(
        arb_struct *p, int *hits_boundary, int *error,
        myquad_t q, const arb_t trust_radius);

void quad_estimate_improvement(arb_t d, myquad_t q, const arb_struct *p);
void quad_evaluate_gradient(myquad_t q);
void quad_evaluate_objective(myquad_t q);
int quad_evaluate_newton(myquad_t q);
int quad_evaluate_cauchy(myquad_t q);

#ifdef __cplusplus
}
#endif

#endif
