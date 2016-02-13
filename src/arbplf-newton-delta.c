#include "arbplfhess.h"
#include "runjson.h"

int main(void)
{
    int result;
    json_hom_t hom;

    hom->userdata = newton_delta_query;
    hom->clear = NULL;
    hom->f = arbplf_second_order_run;

    result = run_json_script(hom);
    return result;
}
