#include "flint/flint.h"
#include "arb.h"
#include "arb_mat.h"

#include "arb_calc.h"
#include "arb_vec_calc.h"
#include "rosenbrock.h"

typedef int (* contraction_t) (
        arb_struct *xout, arb_vec_calc_func_t func, void *param,
        const arb_struct *xin, slong n, slong prec);

typedef int (* refinement_t) (
        arb_struct *out, arb_vec_calc_func_t func, void *param,
        const arb_struct *start, slong n,
        slong eval_extra_prec, slong prec);



static int
_objective(arb_struct *vec_out, arb_mat_struct *jac_out,
      const arb_struct *inp, void *param, slong n, slong prec)
{
    const arb_struct *x, *y;

    if (n != 2)
    {
        flint_printf("internal error: only bivariate rosenbrock is implemented\n");
        abort();
    }

    x = inp + 0;
    y = inp + 1;

    if (arb_calc_verbose)
    {
        flint_printf("x:\n");
        arb_printd(inp + 0, 15); flint_printf("\n");
        arb_printd(inp + 1, 15); flint_printf("\n");
        flint_printf("\n");
    }

    rosenbrock_gradient(vec_out + 0, vec_out + 1, x, y, prec);

    if (arb_calc_verbose)
    {
        flint_printf("gradient:\n");
        arb_printd(vec_out + 0, 15); flint_printf("\n");
        arb_printd(vec_out + 1, 15); flint_printf("\n");
        flint_printf("\n");
    }

    rosenbrock_hessian(jac_out, x, y, prec);

    if (arb_calc_verbose)
    {
        flint_printf("hessian:\n");
        arb_mat_printd(jac_out, 15); flint_printf("\n");
        flint_printf("\n");
    }

    return 0;
}


void test_solution_interval(flint_rand_t state)
{
    arb_struct *x_in, *x_out, *x_true;
    int iter;
    slong i, n, prec;

    flint_printf("solution interval....");
    fflush(stdout);

    n = 2;
    x_in = _arb_vec_init(n);
    x_out = _arb_vec_init(n);
    x_true = _arb_vec_init(n);

    for (i = 0; i < n; i++)
        arb_one(x_true + i);

    /* check a solution interval */
    {
        arb_mat_t H0;
        arb_mat_t H1;
        arb_mat_t H;
        arb_mat_t b;
        arb_mat_t X;
        arb_mat_t Y;

        arb_mat_init(X, 2, 1);
        arb_mat_init(Y, 2, 1);

        arb_mat_init(H0, 2, 2);
        arb_set_si(arb_mat_entry(H0, 0, 0), 400000000002);
        arb_set_si(arb_mat_entry(H0, 1, 1), 200);

        arb_mat_init(H1, 2, 2);

        arb_mat_init(H, 2, 2);
        arb_mat_set(H, H0);
        mag_set_ui(arb_radref(arb_mat_entry(H, 0, 0)), 400000000);
        mag_set_ui(arb_radref(arb_mat_entry(H, 0, 1)), 200);
        mag_set_ui(arb_radref(arb_mat_entry(H, 1, 0)), 200);

        arb_mat_init(b, 2, 1);
        arb_set_si(arb_mat_entry(b, 0, 0), -2);

        for (iter = 0; iter < 1000; iter++)
        {
            slong ra, rb, rc;

            prec = 2 + n_randint(state, 1 << n_randint(state, 17));

            arb_mat_solve(X, H, b, prec);

            ra = n_randint(state, 400000000*2 + 1) - 400000000;
            rb = n_randint(state, 200*2 + 1) - 200;
            rc = n_randint(state, 200*2 + 1) - 200;

            arb_add_si(arb_mat_entry(H1, 0, 0),
                       arb_mat_entry(H0, 0, 0), ra, ARF_PREC_EXACT);
            arb_add_si(arb_mat_entry(H1, 0, 1),
                       arb_mat_entry(H0, 0, 1), rb, ARF_PREC_EXACT);
            arb_add_si(arb_mat_entry(H1, 1, 0),
                       arb_mat_entry(H0, 1, 0), rc, ARF_PREC_EXACT);
            arb_add_si(arb_mat_entry(H1, 1, 1),
                       arb_mat_entry(H0, 1, 1), 0, ARF_PREC_EXACT);

            arb_mat_solve(Y, H1, b, prec);

            if (!arb_mat_contains(H, H1) || !arb_mat_contains(H, H0))
            {
                flint_printf("FAIL: internal test error\n");
                abort();
            }

            if (!arb_mat_overlaps(X, Y))
            {
                flint_printf("FAIL: solve overlap\n");
                flint_printf("prec=%wd\n", prec);
                flint_printf("ra=%wd rb=%wd rc=%wd\n", ra, rb, rc);

                flint_printf("X:\n");
                arb_mat_printd(X, 15); flint_printf("\n");

                flint_printf("Y:\n");
                arb_mat_printd(Y, 15); flint_printf("\n");

                flint_printf("\n");

                abort();
            }
            /* if (arb_calc_verbose) */
            if (0)
            {
                flint_printf("debug: ok solutions:\n");
                flint_printf("prec=%wd\n", prec);
                flint_printf("ra=%wd rb=%wd rc=%wd\n", ra, rb, rc);
                arb_mat_printd(X, 15); flint_printf("\n");
                arb_mat_printd(Y, 15); flint_printf("\n");
                flint_printf("\n");
            }
        }

        arb_mat_clear(H);
        arb_mat_clear(b);
        arb_mat_clear(X);
        arb_mat_clear(Y);
    }

    _arb_vec_clear(x_in, n);
    _arb_vec_clear(x_out, n);
    _arb_vec_clear(x_true, n);

    flint_printf("PASS\n");
}


void test_interval_method(contraction_t contraction, refinement_t refinement,
        flint_rand_t state, const char *name)
{
    int iter;

    flint_printf("%s interval method....", name);
    fflush(stdout);

    {
        arb_struct *x_in, *x_out, *x_true;
        int result;
        slong i, n, prec;

        n = 2;
        x_in = _arb_vec_init(n);
        x_out = _arb_vec_init(n);
        x_true = _arb_vec_init(n);

        for (i = 0; i < n; i++)
            arb_one(x_true + i);

        /* hardcoded example near the true root */
        {
            prec = 256;

            arb_set_d(x_in + 0, 1.00001);
            arb_set_d(x_in + 1, 0.99999);

            mag_set_d(arb_radref(x_in + 0), 0.0001);
            mag_set_d(arb_radref(x_in + 1), 0.0001);

            result = refinement(x_out, _objective, NULL, x_in, n, 0, prec);

            flint_printf("input:\n");
            arb_printd(x_in + 0, 15); flint_printf("\n");
            arb_printd(x_in + 1, 15); flint_printf("\n");
            flint_printf("\n");

            flint_printf("output:\n");
            arb_printd(x_out + 0, 15); flint_printf("\n");
            arb_printd(x_out + 1, 15); flint_printf("\n");
            flint_printf("\n");
        }

        /* example found by random testing */
        {
            prec = 256;

            arb_set_d(x_in + 0, 0.0);
            arb_set_d(x_in + 1, 4);

            mag_set_d(arb_radref(x_in + 0), 0.5);
            mag_set_d(arb_radref(x_in + 1), 2);

            flint_printf("hardcoded starting point:\n");
            arb_printd(x_in + 0, 15); flint_printf("\n");
            arb_printd(x_in + 1, 15); flint_printf("\n");

            /* one iteration */
            result = contraction(x_out, _objective, NULL, x_in, n, prec);

            if (_arb_vec_contains(x_in, x_out, n))
            {
                if (!_arb_vec_contains(x_in, x_true, n))
                {
                    flint_printf("FAIL: unexpected containment\n");

                    flint_printf("input:\n");
                    arb_printd(x_in + 0, 15); flint_printf("\n");
                    arb_printd(x_in + 1, 15); flint_printf("\n");

                    flint_printf("output:\n");
                    arb_printd(x_out + 0, 15); flint_printf("\n");
                    arb_printd(x_out + 1, 15); flint_printf("\n");

                    /* one newton iteration */
                    result = contraction(x_in, _objective, NULL, x_out, n, prec);
                    flint_printf("output:\n");
                    arb_printd(x_out + 0, 15); flint_printf("\n");
                    arb_printd(x_out + 1, 15); flint_printf("\n");
                    flint_printf("\n");

                    flint_printf("output 2:\n");
                    arb_printd(x_in + 0, 15); flint_printf("\n");
                    arb_printd(x_in + 1, 15); flint_printf("\n");
                    flint_printf("\n");

                    abort();
                }
            }

            result = refinement(x_out, _objective, NULL, x_in, n, 0, prec);

            flint_printf("input:\n");
            arb_printd(x_in + 0, 15); flint_printf("\n");
            arb_printd(x_in + 1, 15); flint_printf("\n");
            flint_printf("\n");

            flint_printf("output:\n");
            arb_printd(x_out + 0, 15); flint_printf("\n");
            arb_printd(x_out + 1, 15); flint_printf("\n");
            flint_printf("\n");
        }

        /* second example found by random testing */
        {
            prec = 256;

            arb_set_d(x_in + 0, -0.0001);
            arb_set_d(x_in + 1, 15);

            mag_set_d(arb_radref(x_in + 0), 0.025);
            mag_set_d(arb_radref(x_in + 1), 0.5);

            flint_printf("hardcoded starting point:\n");
            arb_printd(x_in + 0, 15); flint_printf("\n");
            arb_printd(x_in + 1, 15); flint_printf("\n");

            /* one iteration */
            result = contraction(x_out, _objective, NULL, x_in, n, prec);

            if (_arb_vec_contains(x_in, x_out, n))
            {
                if (!_arb_vec_contains(x_in, x_true, n))
                {
                    flint_printf("FAIL: unexpected containment\n");

                    flint_printf("input:\n");
                    arb_printd(x_in + 0, 15); flint_printf("\n");
                    arb_printd(x_in + 1, 15); flint_printf("\n");

                    flint_printf("output:\n");
                    arb_printd(x_out + 0, 15); flint_printf("\n");
                    arb_printd(x_out + 1, 15); flint_printf("\n");

                    /* one newton iteration */
                    result = contraction(x_in, _objective, NULL, x_out, n, prec);
                    flint_printf("output:\n");
                    arb_printd(x_out + 0, 15); flint_printf("\n");
                    arb_printd(x_out + 1, 15); flint_printf("\n");
                    flint_printf("\n");

                    flint_printf("output 2:\n");
                    arb_printd(x_in + 0, 15); flint_printf("\n");
                    arb_printd(x_in + 1, 15); flint_printf("\n");
                    flint_printf("\n");

                    abort();
                }
            }

            result = refinement(x_out, _objective, NULL, x_in, n, 0, prec);

            flint_printf("input:\n");
            arb_printd(x_in + 0, 15); flint_printf("\n");
            arb_printd(x_in + 1, 15); flint_printf("\n");
            flint_printf("\n");

            flint_printf("output:\n");
            arb_printd(x_out + 0, 15); flint_printf("\n");
            arb_printd(x_out + 1, 15); flint_printf("\n");
            flint_printf("\n");
        }

        /* random testing */
        for (iter = 0; iter < 10000; iter++)
        {
            for (i = 0; i < n; i++)
                arb_randtest(x_in + i, state, 1 + n_randint(state, 200), 10);

            prec = 2 + n_randint(state, 1 << n_randint(state, 17));

            flint_printf("sampled starting point:\n");
            arb_printd(x_in + 0, 15); flint_printf("\n");
            arb_printd(x_in + 1, 15); flint_printf("\n");

            /* one contraction iteration */
            result = contraction(x_out, _objective, NULL, x_in, n, prec);

            if (_arb_vec_contains(x_in, x_out, n))
            {
                if (!_arb_vec_contains(x_in, x_true, n))
                {
                    flint_printf("FAIL: unexpected containment\n");

                    flint_printf("input:\n");
                    arb_printd(x_in + 0, 15); flint_printf("\n");
                    arb_printd(x_in + 1, 15); flint_printf("\n");

                    flint_printf("output:\n");
                    arb_printd(x_out + 0, 15); flint_printf("\n");
                    arb_printd(x_out + 1, 15); flint_printf("\n");

                    abort();
                }
            }

            /* root refinement using multiple iterations */
            result = refinement(x_out, _objective, NULL, x_in, n, 0, prec);

            if (result < -1 || result > 1)
            {
                flint_printf("FAIL: unexpected %s refinement integer result\n",
                        name);
                abort();
            }
        }

        _arb_vec_clear(x_in, n);
        _arb_vec_clear(x_out, n);
        _arb_vec_clear(x_true, n);
    }

    flint_printf("PASS\n");
}


int main(void)
{
    FLINT_TEST_INIT(state);

    arb_calc_verbose = 1;

    test_solution_interval(state);

    test_interval_method(
            _arb_vec_calc_newton_contraction,
            _arb_vec_calc_refine_root_newton,
            state, "newton");

    test_interval_method(
            _arb_vec_calc_krawczyk_contraction,
            _arb_vec_calc_refine_root_krawczyk,
            state, "krawczyk");

    FLINT_TEST_CLEANUP(state);

    return EXIT_SUCCESS;
}
