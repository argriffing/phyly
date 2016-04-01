#include "arb_calc.h"

#include "util.h"
#include "arb_vec_extras.h"
#include "arb_vec_calc_quad.h"

static int
_arb_vec_is_small(arb_struct *v, slong n, const arb_t r, slong prec)
{
    int result;
    arb_t d2, r2;

    arb_init(d2);
    _arb_vec_dot(d2, v, v, n, prec);

    arb_init(r2);
    arb_mul(r2, r, r, prec);

    result = arb_lt(d2, r2);

    arb_clear(d2);
    arb_clear(r2);

    return result;
}

static void
_quad_form(arb_t y, const arb_mat_t A, const arb_struct *x, slong prec)
{
    slong n;
    arb_struct *u;
    n = arb_mat_nrows(A);
    u = _arb_vec_init(n);
    _arb_vec_mul_arb_mat(u, x, A, prec);
    _arb_vec_dot(y, u, x, n, prec);
    _arb_vec_clear(u, n);
}


static void
_trust_region_intersections(arb_t ta, arb_t tb,
        const arb_struct *z, const arb_struct *d, slong n,
        const arb_t r, slong prec)
{
    /* 
     * solve ||z + t*d|| == r for scalar t,
     * given vector z, vector d, and scalar r.
     */
    arb_t a, b, sqrt_discriminant;

    /* dot(d, d) */
    arb_init(a);
    _arb_vec_dot(a, d, d, n, prec);

    /* 2*dot(z, d) */
    arb_init(b);
    _arb_vec_dot(b, z, d, n, prec);
    arb_mul_2exp_si(b, b, 1);

    /* sqrt(b*b - 4*a*c) */
    arb_init(sqrt_discriminant);
    {
        arb_t r2, c, b2, four_ac;

        arb_init(r2);
        arb_mul(r2, r, r, prec);

        arb_init(c);
        _arb_vec_dot(c, z, z, n, prec);
        arb_sub(c, c, r2, prec);

        arb_init(b2);
        arb_mul(b2, b, b, prec);

        arb_init(four_ac);
        arb_mul(four_ac, a, c, prec);
        arb_mul_2exp_si(four_ac, four_ac, 2);

        arb_sub(sqrt_discriminant, b2, four_ac, prec);
        arb_sqrt(sqrt_discriminant, sqrt_discriminant, prec);

        arb_clear(r2);
        arb_clear(c);
        arb_clear(b2);
        arb_clear(four_ac);
    }

    /* ta = (-b - sqrt(b*b - 4*a*c)) / (2*a) */
    /* tb = (-b + sqrt(b*b - 4*a*c)) / (2*a) */
    {
        arb_t two_a;

        arb_init(two_a);
        arb_mul_2exp_si(two_a, a, 1);

        arb_neg(ta, b);
        arb_sub(ta, ta, sqrt_discriminant, prec);
        arb_div(ta, ta, two_a, prec);

        arb_neg(tb, b);
        arb_add(tb, tb, sqrt_discriminant, prec);
        arb_div(tb, tb, two_a, prec);

        arb_clear(two_a);
    }

    arb_clear(a);
    arb_clear(b);
    arb_clear(sqrt_discriminant);
}

void
quad_printd(const myquad_t q, slong digits)
{
    if (q->x)
    {
        flint_printf("x:\n");
        _arb_vec_printd(q->x, q->n, digits);
    }
    if (q->y)
    {
        flint_printf("y:\n");
        arb_printd(q->y, digits);
        flint_printf("\n");
    }
    if (q->g)
    {
        flint_printf("g:\n");
        _arb_vec_printd(q->g, q->n, digits);
    }
    if (q->h)
    {
        flint_printf("h:\n");
        arb_mat_printd(q->h, digits);
    }
}

void
quad_init(myquad_t q, arb_vec_calc_f_t f,
        arb_struct *x, void *param, slong n, slong prec)
{
    q->n = n;
    q->y = NULL;
    q->g = NULL;
    q->h = NULL;
    q->p_cauchy = NULL;
    q->p_newton = NULL;
    q->f = f;
    q->param = param;
    q->prec = prec;

    /* copy x */
    q->x = _arb_vec_init(n);
    _arb_vec_set(q->x, x, n);
}

void
quad_clear(myquad_t q)
{
    if (q->x)
    {
        _arb_vec_clear(q->x, q->n);
        q->x = NULL;
    }
    if (q->y)
    {
        arb_clear(q->y);
        flint_free(q->y);
        q->y = NULL;
    }
    if (q->g)
    {
        _arb_vec_clear(q->g, q->n);
        q->g = NULL;
    }
    if (q->h)
    {
        arb_mat_clear(q->h);
        flint_free(q->h);
        q->h = NULL;
    }
    if (q->p_cauchy)
    {
        _arb_vec_clear(q->p_cauchy, q->n);
        q->p_cauchy = NULL;
    }
    if (q->p_newton)
    {
        _arb_vec_clear(q->p_newton, q->n);
        q->p_newton = NULL;
    }
    q->f = NULL;
    q->param = NULL;
    q->prec = -1;
    q->n = -1;
}

void
quad_init_set(myquad_t b, const myquad_t a)
{
    b->x = NULL;
    b->y = NULL;
    b->g = NULL;
    b->h = NULL;
    b->p_cauchy = NULL;
    b->p_newton = NULL;
    b->n = a->n;
    quad_set(b, a);
}

arb_struct *
quad_alloc_y(myquad_t q)
{
    q->y = flint_malloc(sizeof(arb_struct));
    arb_init(q->y);
    return q->y;
}

arb_struct *
quad_alloc_g(myquad_t q)
{
    q->g = _arb_vec_init(q->n);
    return q->g;
}

arb_mat_struct *
quad_alloc_h(myquad_t q)
{
    q->h = flint_malloc(sizeof(arb_mat_struct));
    arb_mat_init(q->h, q->n, q->n);
    return q->h;
}

void
quad_set(myquad_t b, const myquad_t a)
{
    if (b->n != a->n)
    {
        flint_printf("internal error: incompatible quad shapes\n");
        abort();
    }

    /*
     * Completely clear the existing state. This simplifies the code
     * at the cost of redundant memory allocations/deallocations,
     */
    quad_clear(b);

    b->n = a->n;
    b->f = a->f;
    b->param = a->param;
    b->prec = a->prec;

    if (a->x)
    {
        b->x = _arb_vec_init(a->n);
        _arb_vec_set(b->x, a->x, a->n);
    }
    if (a->y)
    {
        arb_set(quad_alloc_y(b), a->y);
    }
    if (a->g)
    {
        _arb_vec_set(quad_alloc_g(b), a->g, a->n);
    }
    if (a->h)
    {
        arb_mat_set(quad_alloc_h(b), a->h);
    }
    if (a->p_newton)
    {
        b->p_newton = _arb_vec_init(a->n);
        _arb_vec_set(b->p_newton, a->p_newton, a->n);
    }
    if (a->p_cauchy)
    {
        b->p_cauchy = _arb_vec_init(a->n);
        _arb_vec_set(b->p_cauchy, a->p_cauchy, a->n);
    }
}



/* todo: move this to an optimization-specific file? */
void
_minimize_dogleg(myquad_t q_opt, myquad_t q_initial,
        const arb_t initial_radius, const arb_t max_radius, slong maxiter)
{
    slong iter;
    slong n, prec;
    int error, hits_boundary;
    arb_struct *p, *x;
    quad_struct quads[2];
    quad_struct *q_curr, *q_next, *q_tmp;
    arb_t eta, r_curr, rho;

    arb_init(eta);
    arb_set_d(eta, 0.15);

    n = q_initial->n;
    prec = q_initial->prec;

    q_curr = quads + 0;
    quad_init_set(q_curr, q_initial);

    q_next = quads + 1;
    quad_init_set(q_next, q_initial);

    p = _arb_vec_init(n);
    x = _arb_vec_init(n);

    arb_init(r_curr);
    arb_set(r_curr, initial_radius);

    arb_init(rho);

    iter = 0;
    while (1)
    {
        _solve_dogleg_subproblem(p, &hits_boundary, &error, q_curr, r_curr);
        if (error)
        {
            /*
             * Maybe the precision is not high enough,
             * or maybe the guess is not near a local minimum.
             */
            if (arb_calc_verbose)
            {
                flint_printf("debug: an error was reported "
                             "in the solution of the dogleg subproblem\n");
            }
            goto finish;
        }
        _arb_vec_add(x, q_curr->x, p, n, prec);

        /* initialize the local model at the proposed point */
        quad_clear(q_next);
        quad_init(q_next, q_curr->f, x, q_curr->param, n, prec);

        if (q_next->n != n)
        {
            flint_printf("internal error: invalid dimension\n");
            abort();
        }

        if (q_curr->n != n)
        {
            flint_printf("internal error: invalid dimension\n");
            abort();
        }

        /* get the ratio of the observed to expected improvement */
        {
            int expectation_ok;
            arb_t observed, expected;

            arb_init(observed);
            arb_init(expected);

            /* expected improvement according to the local model */
            quad_estimate_improvement(expected, q_curr, p);
            expectation_ok = arb_is_positive(expected);

            if (expectation_ok)
            {
                /* observed improvement when the objectives are evaluated */
                if (!q_next->y)
                    quad_evaluate_objective(q_next);
                if (!q_curr->y)
                    quad_evaluate_objective(q_curr);
                arb_sub(observed, q_curr->y, q_next->y, prec);

                /* set rho to the ratio of observed to expected improvement */
                arb_div(rho, observed, expected, prec);
            }

            arb_clear(observed);
            arb_clear(expected);

            if (!expectation_ok)
            {
                /* Maybe the precision is not high enough. */
                if (arb_calc_verbose)
                {
                    flint_printf("debug: estimated improvement "
                                 "is not positive\n");
                }
                goto finish;
            }
        }

        /* update the trust region radius */
        {
            arb_t tmp;
            arb_init(tmp);

            /* maybe contract the radius */
            arb_set_d(tmp, 0.25);
            if (arb_lt(rho, tmp))
            {
                if (arb_calc_verbose)
                {
                    flint_printf("contracting trust radius\n");
                    arb_printd(r_curr, 15);
                    flint_printf("\n");
                }

                arb_mul_2exp_si(r_curr, r_curr, -2);

                if (arb_calc_verbose)
                {
                    arb_printd(r_curr, 15);
                    flint_printf("\n");
                }
            }

            /* maybe expand the radius */
            arb_set_d(tmp, 0.75);
            if (arb_gt(rho, tmp) && hits_boundary)
            {
                if (arb_calc_verbose)
                {
                    flint_printf("expanding trust radius\n");
                    arb_printd(r_curr, 15);
                    flint_printf("\n");
                }

                arb_mul_2exp_si(r_curr, r_curr, 1);
                if (arb_gt(r_curr, max_radius))
                {
                    arb_set(r_curr, max_radius);
                }

                if (arb_calc_verbose)
                {
                    arb_printd(r_curr, 15);
                    flint_printf("\n");
                }
            }

            arb_clear(tmp);
        }

        /* maybe accept the proposed step */
        if (arb_gt(rho, eta))
        {
            /* swap curr and next local models */
            q_tmp = q_curr;
            q_curr = q_next;
            q_next = q_tmp;
        }

        /* increase iteration count */
        iter++;

        /* finish if the gradient interval contains the zero vector */
        {
            if (!q_curr->g)
                quad_evaluate_gradient(q_curr);

            if (_arb_vec_contains_zero(q_curr->g, n))
            {
                if (arb_calc_verbose)
                {
                    flint_printf("debug: gradient contains zero\n");
                }
                goto finish;
            }
        }

        /* finish if the iteration limit has been reached */
        if (iter >= maxiter)
        {
            if (arb_calc_verbose)
            {
                flint_printf("debug: hit iteration limit\n");
            }
            goto finish;
        }
    }

finish:

    if (arb_calc_verbose)
    {
        flint_printf("debug: %d minimize_dogleg iterations\n", iter);
    }

    quad_set(q_opt, q_curr);

    arb_clear(r_curr);
    arb_clear(rho);
    arb_clear(eta);

    _arb_vec_clear(p, n);
    _arb_vec_clear(x, n);
    quad_clear(q_curr);
    quad_clear(q_next);
}


/* todo: move this to an optimization-specific file? */
void
_solve_dogleg_subproblem(
        arb_struct *p, int *hits_boundary, int *error,
        myquad_t q, const arb_t trust_radius)
{
    int invertible;

    *error = 0;
    *hits_boundary = 0;

    /* check if the newton point is within the trust region */
    if (!q->p_newton)
    {
        invertible = quad_evaluate_newton(q);
        if (!invertible)
        {
            *error = -1;
            return;
        }
    }
    if (_arb_vec_is_small(q->p_newton, q->n, trust_radius, q->prec))
    {
        _arb_vec_set(p, q->p_newton, q->n);
        if (arb_calc_verbose)
        {
            flint_printf("using newton point\n");
            /* _arb_vec_printd(p, q->n, 15); */
        }
        return;
    }

    /*
     * The solution is on the boundary, and lies
     * either between the origin and the cauchy point,
     * or between the cauchy point and the newton point.
     */
    *hits_boundary = 1;

    /* check if the cauchy point is within the trust region */
    if (!q->p_cauchy)
    {
        invertible = quad_evaluate_cauchy(q);
        if (!invertible)
        {
            *error = -1;
            return;
        }
    }
    if (_arb_vec_is_small(q->p_cauchy, q->n, trust_radius, q->prec))
    {
        arb_t ta, tb;
        arb_struct *v;

        arb_init(ta);
        arb_init(tb);
        v = _arb_vec_init(q->n);

        /* get the vector from the cauchy point to the newton point */
        _arb_vec_sub(v, q->p_newton, q->p_cauchy, q->n, q->prec);

        _trust_region_intersections(ta, tb,
                q->p_cauchy, v, q->n, trust_radius, q->prec);

        /* p <- cauchy + t * (newton - cauchy) */
        _arb_vec_set(p, q->p_cauchy, q->n);
        _arb_vec_scalar_addmul(p, v, q->n, tb, q->prec);

        _arb_vec_clear(v, q->n);
        arb_clear(ta);
        arb_clear(tb);

        if (arb_calc_verbose)
        {
            flint_printf("interpolating between cauchy and newton points\n");
            /* _arb_vec_printd(p, q->n, 15); */
        }

        return;
    }

    /* the solution is in the direction of the cauchy point */
    /* p <- cauchy * (trust_radius / cauchy_distance) */
    {
        arb_t d, d2;

        arb_init(d2);
        _arb_vec_dot(d2, q->p_cauchy, q->p_cauchy, q->n, q->prec);

        arb_init(d);
        arb_sqrt(d, d2, q->prec);

        _arb_vec_scalar_mul(p, q->p_cauchy, q->n, trust_radius, q->prec);
        _arb_vec_scalar_div(p, p, q->n, d, q->prec);

        arb_clear(d2);
        arb_clear(d);

        if (arb_calc_verbose)
        {
            flint_printf("using a point in the cauchy direction\n");
            /* _arb_vec_printd(p, q->n, 15); */
        }
    }
}

void
quad_estimate_improvement(arb_t d, myquad_t q, const arb_struct *p)
{
    /* request gradient and hessian of the nonlinear function */
    {
        arb_struct *y_req = NULL;
        arb_struct *g_req = NULL;
        arb_mat_struct *h_req = NULL;
        if (!q->g)
        {
            g_req = quad_alloc_g(q);
        }
        if (!q->h)
        {
            h_req = quad_alloc_h(q);
        }
        q->f(y_req, g_req, h_req, q->x, q->param, q->n, q->prec);
    }

    /* estimate the improvement according to the local model */
    {
        arb_t a, b;

        arb_init(a);
        _arb_vec_dot(a, p, q->g, q->n, q->prec);

        arb_init(b);
        _quad_form(b, q->h, p, q->prec);
        arb_mul_2exp_si(b, b, -1);

        arb_add(d, a, b, q->prec);
        arb_neg(d, d);

        arb_clear(a);
        arb_clear(b);
    }
}

void
quad_evaluate_gradient(myquad_t q)
{
    if (q->g)
    {
        flint_fprintf(stderr, "internal error: redundant gradient eval\n");
        abort();
    }

    /* request only the gradient, not the objective or the hessian */
    q->f(NULL, quad_alloc_g(q), NULL, q->x, q->param, q->n, q->prec);
}

void
quad_evaluate_objective(myquad_t q)
{
    if (q->y)
    {
        flint_fprintf(stderr, "internal error: redundant objective eval\n");
        abort();
    }

    /* request only the objective, not the gradient or hessian */
    q->f(quad_alloc_y(q), NULL, NULL, q->x, q->param, q->n, q->prec);
}

int
quad_evaluate_newton(myquad_t q)
{
    /* allocate the newton offset */
    if (q->p_newton)
    {
        flint_fprintf(stderr, "internal error: redundant newton eval\n");
        abort();
    }
    q->p_newton = _arb_vec_init(q->n);

    /* request gradient and hessian of the nonlinear function */
    {
        arb_struct *y_req = NULL;
        arb_struct *g_req = NULL;
        arb_mat_struct *h_req = NULL;
        if (!q->g)
        {
            g_req = quad_alloc_g(q);
        }
        if (!q->h)
        {
            h_req = quad_alloc_h(q);
        }
        q->f(y_req, g_req, h_req, q->x, q->param, q->n, q->prec);
    }

    /* compute the newton offset using the taylor terms */
    {
        int invertible;

        invertible = _arb_mat_solve_arb_vec(q->p_newton, q->h, q->g, q->prec);
        _arb_vec_neg(q->p_newton, q->p_newton, q->n);

        if (arb_calc_verbose)
        {
            flint_printf("newton point computation:\n");
            flint_printf("h in:\n");
            arb_mat_printd(q->h, 15);
            flint_printf("g in:\n");
            _arb_vec_printd(q->g, q->n, 15);
            flint_printf("newton point out:\n");
            _arb_vec_printd(q->p_newton, q->n, 15);
        }

        if (!invertible)
        {
            _arb_vec_indeterminate(q->p_newton, q->n);
        }

        return invertible;
    }
}

int
quad_evaluate_cauchy(myquad_t q)
{
    /* allocate the cauchy offset */
    if (q->p_cauchy)
    {
        flint_fprintf(stderr, "internal error: redundant cauchy eval\n");
        abort();
    }
    q->p_cauchy = _arb_vec_init(q->n);

    /* request gradient and hessian of the nonlinear function */
    {
        arb_struct *y_req = NULL;
        arb_struct *g_req = NULL;
        arb_mat_struct *h_req = NULL;
        if (!q->g)
        {
            g_req = quad_alloc_g(q);
        }
        if (!q->h)
        {
            h_req = quad_alloc_h(q);
        }
        q->f(y_req, g_req, h_req, q->x, q->param, q->n, q->prec);
    }

    /* compute the cauchy offset using the taylor terms */
    {
        int invertible;
        arb_t a, b;

        /* the numerator is a = dot(g, g) */
        arb_init(a);
        _arb_vec_dot(a, q->g, q->g, q->n, q->prec);

        /* the denominator is b = dot(g, H, g) */
        arb_init(b);
        _quad_form(b, q->h, q->g, q->prec);
        invertible = !arb_contains_zero(b);

        /* cauchy offset is -(a/b)*g */
        if (invertible)
        {
            _arb_vec_scalar_mul(q->p_cauchy, q->g, q->n, a, q->prec);
            _arb_vec_scalar_div(q->p_cauchy, q->p_cauchy, q->n, b, q->prec);
            _arb_vec_neg(q->p_cauchy, q->p_cauchy, q->n);
        }
        else
        {
            _arb_vec_indeterminate(q->p_cauchy, q->n);
        }

        arb_clear(a);
        arb_clear(b);

        return invertible;
    }
}
