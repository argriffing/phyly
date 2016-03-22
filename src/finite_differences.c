#include "flint/flint.h"

#include "arb.h"
#include "arb_mat.h"

#include "util.h"
#include "finite_differences.h"


void gradient_param_init(gradient_param_t g,
        multivariate_function_t func, void *param, const arb_t delta)
{
    g->func = func;
    g->param = param;
    _arb_init_set(g->delta, delta);
}

void gradient_param_clear(gradient_param_t g)
{
    g->func = NULL;
    g->param = NULL;
    arb_clear(g->delta);
}


/* 'p->func' maps R^N -> R */
/* This follows the 'multivariate_vector_function_t' interface. */
void finite_differences_gradient(
        arb_struct *gradient, const arb_struct *x,
        gradient_param_t p,
        slong n, slong k, slong prec)
{
    slong i;
    arb_t y;
    arb_t yprime;
    arb_struct *xprime;

    if (n != k)
    {
        flint_printf("internal error: gradient dimensions mismatch\n");
        abort();
    }

    arb_init(y);
    arb_init(yprime);
    xprime = _arb_vec_init(n);

    p->func(y, x, p->param, n, prec);

    for (i = 0; i < n; i++)
    {
        _arb_vec_set(xprime, x, n);
        arb_add(xprime + i, xprime + i, p->delta, prec);
        p->func(yprime, xprime, p->param, n, prec);
        arb_sub(gradient + i, yprime, y, prec);
        arb_div(gradient + i, gradient + i, p->delta, prec);
    }

    arb_clear(y);
    arb_clear(yprime);
    _arb_vec_clear(xprime, n);
}


/* This is the Jacobian matrix, not the Jacobian determinant. */
/* 'func' maps R^n -> R^k */
/* J has shape (k, n) */
void finite_differences_jacobian(arb_mat_t J,
        multivariate_vector_function_t func, void *param,
        const arb_struct *x, slong n, slong k, const arb_t delta, slong prec)
{
    slong i, j;
    arb_struct *y;
    arb_struct *yprime;
    arb_struct *xprime;
    arb_struct *tmp;

    y = _arb_vec_init(k);
    yprime = _arb_vec_init(k);
    xprime = _arb_vec_init(n);
    tmp = _arb_vec_init(k);

    func(y, x, param, n, k, prec);

    for (i = 0; i < n; i++)
    {
        _arb_vec_set(xprime, x, n);
        arb_add(xprime + i, xprime + i, delta, prec);
        func(yprime, xprime, param, n, k, prec);
        _arb_vec_sub(tmp, yprime, y, k, prec);
        for (j = 0; j < k; j++)
        {
            arb_div(arb_mat_entry(J, j, i), tmp + j, delta, prec);
        }
    }

    _arb_vec_clear(y, k);
    _arb_vec_clear(yprime, k);
    _arb_vec_clear(xprime, n);
    _arb_vec_clear(tmp, k);
}


void finite_differences_hessian(arb_mat_t H, multivariate_function_t func,
        void *param, const arb_struct *x, slong n,
        const arb_t delta, slong prec)
{
    gradient_param_t g;
    gradient_param_init(g, func, param, delta);
    g->func = func;
    g->param = param;
    finite_differences_jacobian(
            H, finite_differences_gradient, g,
            x, n, n, delta, prec);
    gradient_param_clear(g);
}
