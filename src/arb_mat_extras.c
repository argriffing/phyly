#include "flint/flint.h"

#include "arb.h"
#include "arb_mat.h"

#include "util.h"
#include "arb_mat_extras.h"

arb_mat_struct *
_arb_mat_vec_init(slong nrows, slong ncols, slong len)
{
    slong i;
    arb_mat_struct *vec = flint_malloc(len * sizeof(arb_mat_struct));
    for (i = 0; i < len; i++)
    {
        arb_mat_init(vec + i, nrows, ncols);
    }
    return vec;
}

void
_arb_mat_vec_clear(arb_mat_struct *vec, slong len)
{
    slong i;
    if (vec)
    {
        for (i = 0; i < len; i++)
        {
            arb_mat_clear(vec + i);
        }
        flint_free(vec);
    }
}
