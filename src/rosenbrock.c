#include "rosenbrock.h"

void
rosenbrock_objective(arb_t z, const arb_t x, const arb_t y, slong prec)
{
    arb_t x2, xcompl, xcompl2;

    /* x^2 */
    arb_init(x2);
    arb_mul(x2, x, x, prec);

    /* (1 - x) */
    arb_init(xcompl);
    arb_sub_si(xcompl, x, 1, prec);
    arb_neg(xcompl, xcompl);

    /* (1 - x)^2 */
    arb_init(xcompl2);
    arb_mul(xcompl2, xcompl, xcompl, prec);

    /* (y - x^2)^2 * 100 + (1 - x)^2 */
    arb_sub(z, y, x2, prec);
    arb_mul(z, z, z, prec);
    arb_mul_si(z, z, 100, prec);
    arb_add(z, z, xcompl2, prec);

    arb_clear(x2);
    arb_clear(xcompl);
    arb_clear(xcompl2);
}

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

    /* dy = 200 (y - x^2) */
    arb_sub(dy, y, x2, prec);
    arb_mul_si(dy, dy, 200, prec);

    /* dx = -2 x dy - 2 (1 - x) */
    /*    = -2 (x dy + (1 - x)) */
    arb_set(dx, xcompl);
    arb_addmul(dx, x, dy, prec);
    arb_mul_si(dx, dx, -2, prec);

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
