#include "dummy_module.h"

int main(void)
{
    /*
    json_hom_t hom;
    hom->userdata = NULL;
    hom->clear = NULL;
    hom->f = run;
    int result = run_json_script(hom);

    flint_cleanup();
    return result;
    */
    int i;
    i = 0;
    return add_two_ints(i, i);
}
