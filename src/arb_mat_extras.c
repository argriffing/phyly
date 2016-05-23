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

int
_arb_mat_col_is_constant(const arb_mat_t A, slong j)
{
    slong i;
    arb_srcptr ref;
    if (arb_mat_is_empty(A))
        return 1;
    ref = arb_mat_entry(A, 0, j);
    if (!arb_is_exact(ref))
        return 0;
    for (i = 0; i < arb_mat_nrows(A); i++)
    {
        if (!arb_equal(arb_mat_entry(A, i, j), ref))
            return 0;
    }
    return 1;
}
