#include "flint/flint.h"
#include "flint/fmpz.h"

int main(void)
{
    int iter;
    FLINT_TEST_INIT(state);

    flint_printf("dummy_test....");
    fflush(stdout);

    for (iter = 0; iter < 1000; iter++)
    {
        fmpz_t x;
        fmpz_init_set_ui(x, 42);
        if (!fmpz_equal(x, x))
        {
            flint_printf("FAIL: (fmpz is not equal to itself)\n");
            abort();
        }
        fmpz_clear(x);
    }

    FLINT_TEST_CLEANUP(state);

    flint_printf("PASS\n");
    return 0;
}
