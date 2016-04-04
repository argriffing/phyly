#ifndef MODEL_H
#define MODEL_H

#include "arb.h"
#include "arb_mat.h"

#include "csr_graph.h"


#ifdef __cplusplus
extern "C" {
#endif


typedef struct
{
    double *data;
    int s;
    int r;
    int c;
} pmat_struct;
typedef pmat_struct pmat_t[1];

int pmat_nsites(const pmat_t mat);
int pmat_nrows(const pmat_t mat);
int pmat_ncols(const pmat_t mat);
double * pmat_entry(pmat_t mat, int i, int j, int k);
const double * pmat_srcentry(const pmat_t mat, int i, int j, int k);
void pmat_pre_init(pmat_t mat);
void pmat_init(pmat_t mat, int s, int r, int c);
void pmat_clear(pmat_t mat);
void pmat_update_base_node_vectors(
        arb_mat_struct *base_node_vectors, const pmat_t p, slong site,
        int use_equilibrium_root_prior, arb_struct *equilibrium,
        slong root_node_index, slong prec);


typedef struct
{
    csr_graph_t g;
    arb_mat_t mat;
    pmat_t p;
    int root_node_index;
    int *preorder;
    int use_equilibrium_root_prior;
    int use_equilibrium_rate_divisor;
    arb_t rate_divisor;
    double *edge_rate_coefficients;
    csr_edge_mapper_t edge_map;
} model_and_data_struct;
typedef model_and_data_struct model_and_data_t[1];

void model_and_data_init(model_and_data_t m);
void model_and_data_clear(model_and_data_t m);

/* Update using a new level of precision. */
void
_update_rate_matrix_and_equilibrium(
        arb_mat_t rate_matrix,
        arb_struct *equilibrium,
        arb_t rate_divisor,
        int use_equilibrium_root_prior,
        int use_equilibrium_rate_divisor,
        const arb_mat_t mat,
        slong prec);

#ifdef __cplusplus
}
#endif

#endif
