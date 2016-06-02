#include "flint/flint.h"
#include "arb.h"
#include "arb_calc.h"

#include "arb_vec_extras.h"
#include "gamma_discretization.h"

int
main()
{
    slong k;
    slong n = 4;
    double shape = 0.5;
    //double shape = 0.19242344607262146239;
    arb_t s;
    arb_struct *rates;
    slong prec = 4;

    arb_init(s);
    arb_set_d(s, shape);
    rates = _arb_vec_init(n);

    while (1)
    {
        slong worstprec;
        gamma_rates(rates, n, s, prec);
        for (k = 0; k < n; k++)
        {
            slong p = arb_rel_accuracy_bits(rates + k);
            if ((k == 0) || p < worstprec)
                worstprec = p;
        }
        if (worstprec > 53)
            break;
        prec <<= 1;
    }

    flint_printf("rates:\n");
    for (k = 0; k < n; k++)
    {
        arb_printd(rates + k, 15);
        flint_printf("\n");
    }

    _arb_vec_clear(rates, n);

    return 0;
}
