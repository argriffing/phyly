#include "arbplfem.h"
#include "runjson.h"

int main(void)
{
    int result;
    json_hom_t hom;

    hom->userdata = NULL;
    hom->clear = NULL;
    hom->f = arbplf_em_update_run;

    result = run_json_script(hom);
    return result;
}
