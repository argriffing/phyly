#include "flint/flint.h"
#include "arb.h"
#include "arb_mat.h"

#include "arb_calc.h"
#include "arb_vec_calc.h"


void
rosen_grad(arb_t dx, arb_t dy, const arb_t x, const arb_t y, slong prec)
{
    arb_t x2;
    arb_t xcompl;

    arb_init(x2);
    arb_init(xcompl);

    arb_mul(x2, x, x, prec);
    arb_sub_si(xcompl, x, 1, prec);
    arb_neg(xcompl, xcompl);

    arb_sub(dy, y, x2, prec);
    arb_mul(dy, dy, x, prec);
    arb_mul_si(dy, dy, 200, prec);

    arb_mul_si(dx, dy, -2, prec);
    arb_submul_si(dx, xcompl, 2, prec);

    arb_clear(x2);
    arb_clear(xcompl);
}

void
rosen_hess(arb_mat_t H, const arb_t x, const arb_t y, slong prec)
{
    arb_t x2;
    arb_struct *p;

    arb_init(x2);

    arb_mul(x2, x, x, prec);

    p = arb_mat_entry(H, 0, 0);
    arb_set_si(p, 2);
    arb_addmul_si(p, x2, 1200, prec);
    arb_submul_si(p, y, 400, prec);

    p = arb_mat_entry(H, 0, 1);
    arb_mul_si(p, x, -400, prec);

    p = arb_mat_entry(H, 1, 0);
    arb_mul_si(p, x, -400, prec);

    p = arb_mat_entry(H, 1, 1);
    arb_set_si(p, 200);

    arb_clear(x2);
}

static int
_objective(arb_struct *vec_out, arb_mat_struct *jac_out,
      const arb_struct *inp, void *param, slong n, slong prec)
{
    const arb_struct *x, *y;

    x = inp + 0;
    y = inp + 1;

    rosen_grad(vec_out + 0, vec_out + 1, x, y, prec);
    rosen_hess(jac_out, x, y, prec);

    flint_printf("vec:\n");
    arb_printd(vec_out + 0, 15); flint_printf("\n");
    arb_printd(vec_out + 1, 15); flint_printf("\n");
    flint_printf("\n");

    flint_printf("jacobian:\n");
    arb_mat_printd(jac_out, 15); flint_printf("\n");
    flint_printf("\n");

    return 0;
}

int main(void)
{
    // int iter;
    // FLINT_TEST_INIT(state);

    flint_printf("newton....");
    fflush(stdout);

    {
        int refinement_result;
        slong n, i, prec;
        arb_struct *vec_in, *vec_out;

        arb_calc_verbose = 1;

        prec = 256;
        n = 2;

        vec_in = _arb_vec_init(n);
        vec_out = _arb_vec_init(n);

        arb_set_d(vec_in + 0, 1.00001);
        arb_set_d(vec_in + 1, 0.99999);

        mag_set_d(arb_radref(vec_in + 0), 0.0001);
        mag_set_d(arb_radref(vec_in + 1), 0.0001);

        refinement_result = arb_vec_calc_refine_root_newton(
                vec_out, _objective, NULL, vec_in, n, 0, prec);

        flint_printf("input:\n");
        arb_printd(vec_in + 0, 15); flint_printf("\n");
        arb_printd(vec_in + 1, 15); flint_printf("\n");
        flint_printf("\n");

        flint_printf("output:\n");
        arb_printd(vec_out + 0, 15); flint_printf("\n");
        arb_printd(vec_out + 1, 15); flint_printf("\n");
        flint_printf("\n");

        _arb_vec_clear(vec_in, n);
        _arb_vec_clear(vec_out, n);
    }

    // FLINT_TEST_CLEANUP(state);

    flint_printf("PASS\n");
    return 0;
}
