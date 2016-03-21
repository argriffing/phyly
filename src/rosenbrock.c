#include "rosenbrock.h"

void
rosenbrock_gradient(arb_t dx, arb_t dy,
        const arb_t x, const arb_t y, slong prec)
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
rosenbrock_hessian(arb_mat_t H,
        const arb_t x, const arb_t y, slong prec)
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
