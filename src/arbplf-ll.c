#include "arbplfll.h"
#include "runjson.h"

int main(void)
{
    int result;
    json_hom_t hom;

    hom->userdata = NULL;
    hom->clear = NULL;
    hom->f = arbplf_ll;

    result = run_json_script(hom);
    return result;
}
