#include "flint/flint.h"
#include "arb.h"
#include "arb_mat.h"

#include "arb_calc.h"
#include "arb_vec_calc.h"
#include "arb_vec_calc_quad.h"
#include "rosenbrock.h"

/*
 * follows arb_vec_calc_f_t in arb_vec_calc_quad.h
 */
static int
_rosenbrock(arb_struct *y, arb_struct *g, arb_mat_struct *h,
      const arb_struct *x, void *param, slong n, slong prec)
{
    if (n != 2)
    {
        flint_printf("internal error: rosenbrock requires n=2\n");
        abort();
    }

    if (y)
    {
        rosenbrock_objective(y, x+0, x+1, prec);
    }

    if (g)
    {
        rosenbrock_gradient(g+0, g+1, x+0, x+1, prec);
    }

    if (h)
    {
        rosenbrock_hessian(h, x+0, x+1, prec);
    }

    return 0;
}


int main(void)
{
    FLINT_TEST_INIT(state);

    arb_calc_verbose = 1;

    {
        int iter;

        flint_printf("arb_vec_calc_quad....");
        fflush(stdout);

        for (iter = 0; iter < 1; iter++)
        {
            slong i, n, prec;
            myquad_t q_opt, q_initial;
            arb_struct *x_in;

            n = 2;

            x_in = _arb_vec_init(n);

            /* sample a random initial point and working precision */
            /*
            for (i = 0; i < n; i++)
            {
                arb_randtest(x_in + i, state, 1 + n_randint(state, 200), 10);
            }
            prec = 2 + n_randint(state, 1 << n_randint(state, 17));
            */
            arb_set_d(x_in + 0, 1.5);
            arb_set_d(x_in + 1, 0.5);
            prec = 256;

            flint_printf("sampled starting point:\n");
            arb_printd(x_in + 0, 15); flint_printf("\n");
            arb_printd(x_in + 1, 15); flint_printf("\n");

            /* initialize the local model at the random initial point */
            quad_init(q_initial, _rosenbrock, x_in, NULL, n, prec);
            quad_init_set(q_opt, q_initial);

            if (q_initial->n != 2)
            {
                flint_printf("FAIL: q_initial has unexpected n\n");
                abort();
            }

            if (q_opt->n != 2)
            {
                flint_printf("FAIL: q_opt has unexpected n\n");
                abort();
            }

            {
                arb_t r, rmax;
                slong maxiter;

                maxiter = 100;

                arb_init(r);
                // arb_set_d(r, 1.0);
                // arb_set_d(r, 0.01);
                arb_set_d(r, 2.0);

                arb_init(rmax);
                arb_set_d(rmax, 10.0);

                _minimize_dogleg(q_opt, q_initial, r, rmax, maxiter);

                arb_clear(r);
                arb_clear(rmax);
            }

            flint_printf("opt point:\n");
            arb_printd(q_opt->x + 0, 15); flint_printf("\n");
            arb_printd(q_opt->x + 1, 15); flint_printf("\n");

            _arb_vec_clear(x_in, n);
            quad_clear(q_initial);
            quad_clear(q_opt);
        }

        flint_printf("PASS\n");
    }

    FLINT_TEST_CLEANUP(state);

    return EXIT_SUCCESS;
}
