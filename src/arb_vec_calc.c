/*
 * Multivariate interval Newton's method for root refinement.
 *
 * Note that this can be used for optimization by looking for a zero (a root)
 * of the gradient. In that case, the Hessian matrix in the optimization
 * problem is treated like the Jacobian matrix in the root-finding problem.
 */
#include "flint/flint.h"

#include "arb_mat.h"
#include "arb.h"
#include "arf.h"

#include "util.h"
#include "arb_calc.h"
#include "arb_vec_extras.h"
#include "arb_vec_calc.h"

static void
_arb_mat_rad(arb_mat_t res, const arb_mat_t mat)
{
    slong i, j;
    arb_mat_set(res, mat);
    for (i = 0; i < arb_mat_nrows(res); i++)
        for (j = 0; j < arb_mat_ncols(res); j++)
            arf_zero(arb_midref(arb_mat_entry(res, i, j)));
}

static void
_arb_mat_mid(arb_mat_t res, const arb_mat_t mat)
{
    slong i, j;
    arb_mat_set(res, mat);
    for (i = 0; i < arb_mat_nrows(res); i++)
        for (j = 0; j < arb_mat_ncols(res); j++)
            mag_zero(arb_radref(arb_mat_entry(res, i, j)));
}

static void
_arb_mat_submul(arb_mat_t z, const arb_mat_t x, const arb_mat_t y, slong prec)
{
    arb_mat_t xy;
    arb_mat_init(xy, arb_mat_nrows(z), arb_mat_ncols(z));
    arb_mat_mul(xy, x, y, prec);
    arb_mat_sub(z, z, xy, prec);
    arb_mat_clear(xy);
}

static void
_arb_mat_addmul(arb_mat_t z, const arb_mat_t x, const arb_mat_t y, slong prec)
{
    arb_mat_t xy;
    arb_mat_init(xy, arb_mat_nrows(z), arb_mat_ncols(z));
    arb_mat_mul(xy, x, y, prec);
    arb_mat_add(z, z, xy, prec);
    arb_mat_clear(xy);
}

int
_arb_vec_calc_krawczyk_contraction(
        arb_struct *xout, arb_vec_calc_func_t func, void *param,
        const arb_struct *xin, slong n, slong prec)
{
    /* K(X) = y - Y f(y) + (I - Y F'(X))(X - y) */
    /* Y = inverse of jacobian of x, discarding error bounds */
    /* f evaluates the function, and F' is the jacobian */
    int invertible;
    slong i;
    arb_mat_t A;
    arb_mat_t Y;
    arb_mat_t J, Jmid, Jrad;
    arb_mat_t xin_col, xin_col_mid, xin_col_rad;
    arb_mat_t vec_col, vec_col_mid;
    arb_mat_t k_col;
    arb_struct *vec;

    arb_mat_init(xin_col, n, 1);
    arb_mat_init(xin_col_mid, n, 1);
    arb_mat_init(xin_col_rad, n, 1);
    arb_mat_init(vec_col, n, 1);
    arb_mat_init(vec_col_mid, n, 1);
    arb_mat_init(k_col, n, 1);

    arb_mat_init(A, n, n);
    arb_mat_init(Y, n, n);
    arb_mat_init(J, n, n);
    arb_mat_init(Jmid, n, n);
    arb_mat_init(Jrad, n, n);

    vec = _arb_vec_init(n);

    /* evaluate the function and jacobian at the input interval */
    func(vec, J, xin, param, n, prec);

    _arb_mat_mid(Jmid, J);
    _arb_mat_rad(Jrad, J);

    invertible = arb_mat_inv(Y, Jmid, prec);
    if (!invertible)
    {
        if (arb_calc_verbose)
        {
            flint_printf("debug: interval includes singular jacobian\n");
            /* arb_mat_printd(jac, 15); */
        }
        _arb_mat_indeterminate(k_col);
    }
    else
    {
        for (i = 0; i < n; i++)
        {
            arb_set(arb_mat_entry(xin_col, i, 0), xin + i);
            arb_set(arb_mat_entry(vec_col, i, 0), vec + i);
        }

        _arb_mat_mid(xin_col_mid, xin_col);
        _arb_mat_rad(xin_col_rad, xin_col);

        _arb_mat_mid(vec_col_mid, vec_col);

        arb_mat_one(A);
        _arb_mat_submul(A, Y, J, prec);

        arb_mat_set(k_col, xin_col_mid);
        _arb_mat_submul(k_col, Y, vec_col_mid, prec);
        _arb_mat_addmul(k_col, A, xin_col_rad, prec);
    }

    for (i = 0; i < n; i++)
        arb_set(xout + i, arb_mat_entry(k_col, i, 0));

    arb_mat_clear(Y);
    arb_mat_clear(J);
    arb_mat_clear(Jmid);
    arb_mat_clear(Jrad);
    arb_mat_clear(xin_col);
    arb_mat_clear(xin_col_mid);
    arb_mat_clear(xin_col_rad);
    arb_mat_clear(vec_col);
    arb_mat_clear(vec_col_mid);
    arb_mat_clear(k_col);
    _arb_vec_clear(vec, n);

    return invertible;
}

int
arb_vec_calc_newton_delta(
        arb_struct *delta, arb_vec_calc_func_t func, void *param,
        const arb_struct *inp, slong n, slong prec)
{
    arb_struct *vec;
    arb_mat_t jac;
    arb_mat_t x, u;
    slong i;
    int invertible;

    vec = _arb_vec_init(n);
    arb_mat_init(jac, n, n);
    arb_mat_init(x, n, 1);
    arb_mat_init(u, n, 1);

    func(vec, jac, inp, param, n, prec);

    for (i = 0; i < n; i++)
        arb_set(arb_mat_entry(x, i, 0), vec + i);

    invertible = arb_mat_solve(u, jac, x, prec);

    for (i = 0; i < n; i++)
        arb_neg(delta + i, arb_mat_entry(u, i, 0));

    _arb_vec_clear(vec, n);
    arb_mat_clear(jac);
    arb_mat_clear(x);
    arb_mat_clear(u);

    return invertible;
}


int
arb_vec_calc_newton_step(
        arb_struct *xnew, arb_vec_calc_func_t func, void *param,
        const arb_struct *x, slong n, slong prec)
{
    int result;
    arb_struct *delta;

    delta = _arb_vec_init(n);

    result = arb_vec_calc_newton_delta(delta, func, param, x, n, prec);

    if (result)
        _arb_vec_add(xnew, x, delta, n, prec);

    _arb_vec_clear(delta, n);

    return result;
}


int
_arb_vec_calc_newton_contraction(
        arb_struct *xnew, arb_vec_calc_func_t func, void *param,
        const arb_struct *inp, slong n, slong prec)
{
    arb_struct *mid_inp;
    arb_struct *mid_vec, *wide_vec;
    arb_mat_t mid_jac, wide_jac;
    arb_mat_t v, u;
    slong i;
    int result;
    int invertible;

    if (_arb_vec_is_indeterminate(inp, n))
    {
        flint_printf("error: indeterminate initial point\n");
        return 0;
    }

    mid_inp = _arb_vec_init(n);
    mid_vec = _arb_vec_init(n);
    wide_vec = _arb_vec_init(n);

    arb_mat_init(mid_jac, n, n);
    arb_mat_init(wide_jac, n, n);
    arb_mat_init(v, n, 1);
    arb_mat_init(u, n, 1);

    if (arb_calc_verbose)
        flint_printf("debug: computing wide interval...\n");

    func(wide_vec, wide_jac, inp, param, n, prec);

    if (_arb_vec_is_indeterminate(wide_vec, n))
    {
        if (arb_calc_verbose)
        {
            flint_printf("debug: indeterminate function evaluation\n");
            result = 0;
            goto finish;
        }
    }

    if (_arb_mat_is_indeterminate(wide_jac))
    {
        if (arb_calc_verbose)
        {
            flint_printf("debug: indeterminate jacobian\n");
            result = 0;
            goto finish;
        }
    }

    /*
     * Use the evaluation of the Jacobian of f at the interval,
     * but use only the evaluation of f only at the midpoint.
     */

    for (i = 0; i < n; i++)
        arb_set_arf(mid_inp + i, arb_midref(inp + i));

    if (arb_calc_verbose)
        flint_printf("debug: computing mid interval...\n");

    func(mid_vec, mid_jac, mid_inp, param, n, prec);

    for (i = 0; i < n; i++)
        arb_set(arb_mat_entry(v, i, 0), mid_vec + i);

    invertible = arb_mat_solve(u, wide_jac, v, prec);
    if (!invertible)
    {
        if (arb_calc_verbose)
        {
            flint_printf("debug: interval includes singular jacobian\n");
            /* arb_mat_printd(jac, 15); */
        }
        _arb_mat_indeterminate(u);
    }

    if (arb_calc_verbose)
    {
        flint_printf("debug: solved\n");
        arb_mat_printd(wide_jac, 15); flint_printf("\n");
        arb_mat_printd(v, 15); flint_printf("\n");
        flint_printf(" = \n");
        arb_mat_printd(u, 15);
        flint_printf("...\n");
        {
            arb_mat_t w;
            arb_mat_init(w, 2, 1);
            arb_mat_mul(w, wide_jac, u, prec);
            arb_mat_printd(w, 15);
            arb_mat_clear(w);
        }
    }

    for (i = 0; i < n; i++)
        arb_sub(xnew + i, mid_inp + i, arb_mat_entry(u, i, 0), prec);

    result = invertible;

finish:

    _arb_vec_clear(wide_vec, n);
    _arb_vec_clear(mid_vec, n);
    _arb_vec_clear(mid_inp, n);
    arb_mat_clear(wide_jac);
    arb_mat_clear(mid_jac);
    arb_mat_clear(v);
    arb_mat_clear(u);

    return result;
}


/* todo: do not ignore eval_extra_prec */
int
arb_vec_calc_refine_root_newton(
        arb_struct *out, arb_vec_calc_func_t func, void *param,
        const arb_struct *start, slong n,
        slong eval_extra_prec, slong prec)
{
    int iter;
    int result;
    int prev_containment;
    arb_struct *x, *xnext;

    result = 1;

    x = _arb_vec_init(n);
    xnext = _arb_vec_init(n);

    _arb_vec_set(x, start, n);
    prev_containment = 0;
    for (iter = 0; !_arb_vec_can_round(x, n); iter++)
    {
        int ret;
        arb_struct *tmp;

        if (arb_calc_verbose)
            flint_printf("debug: iter %d\n", iter);

        _arb_vec_zero(xnext, n);

        ret = _arb_vec_calc_newton_contraction(xnext, func, param, x, n, prec);

        if (arb_calc_verbose)
        {
            _arb_vec_print(x, n); flint_printf("\n");
            flint_printf("->\n");
            _arb_vec_print(xnext, n); flint_printf("\n");
        }

        /* convergence failure */
        if (!ret) {
            result = 0;
            break;
        }
        if (_arb_vec_is_indeterminate(xnext, n)) {
            if (arb_calc_verbose)
                flint_printf("debug: step is indeterminate\n");
            result = 0;
            break;
        }
        if (_arb_vec_equal(x, xnext, n)) {
            if (arb_calc_verbose)
                flint_printf("debug: step is equal\n");
            result = 0;
            break;
        }
        if (_arb_vec_contains(xnext, x, n)) {
            if (arb_calc_verbose)
                flint_printf("debug: step is an expansion\n");
            result = 0;
            break;
        }
        if (_arb_vec_overlaps(x, xnext, n))
        {
            int i;
            int bad = 0;
            for (i = 0; i < n && !bad; i++)
                if (mag_cmp(arb_radref(x), arb_radref(xnext)) < 0)
                    bad = 1;
            if (bad)
            {
                if (arb_calc_verbose)
                    flint_printf("debug: not all error radii have decreased\n");
                result = 0;
                break;
            }
        }

        /* root exclusion */
        if (!_arb_vec_overlaps(x, xnext, n)) {

            if (arb_calc_verbose)
            {
                flint_printf("debug: root is excluded\n");
                _arb_vec_printd(x, n, 15); flint_printf("\n");
                _arb_vec_printd(xnext, n, 15); flint_printf("\n");
            }

            /*
             * If the previous interation indicated root containment,
             * then subsequent root exclusion implies some kind of bug
             * or derivative calculation error or a misunderstanding
             * of the properties of the interval Newton method.
             */
            if (prev_containment)
            {
                flint_printf("internal error: conflicting newton "
                        "iteration results\n");
                abort();
            }

            result = -1;
            break;
        }

        prev_containment = _arb_vec_contains(x, xnext, n);

        tmp = x;
        x = xnext;
        xnext = tmp;
    }

    if (arb_calc_verbose)
        flint_printf("newton iterations: %d\n", iter+1);

    _arb_vec_set(out, x, n);

    _arb_vec_clear(x, n);
    _arb_vec_clear(xnext, n);

    return result;
}
