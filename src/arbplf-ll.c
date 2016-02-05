#include "flint/flint.h"

#include "arbplf_ll.h"
#include "runjson.h"

int main(void)
{
    json_hom_t hom;
    hom->userdata = NULL;
    hom->clear = NULL;
    hom->f = arbplf_ll;
    int result = run_json_script(hom);

    flint_cleanup();
    return result;
}
