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
        arb_struct *equilibrium, slong root_node_index, slong prec);


typedef struct
{
    double *data;
    int r;
    int c;
} dmat_struct;
typedef dmat_struct dmat_t[1];

int dmat_nrows(const dmat_t mat);
int dmat_ncols(const dmat_t mat);
double * dmat_entry(dmat_t mat, int i, int j);
const double * dmat_srcentry(const dmat_t mat, int i, int j);
void dmat_pre_init(dmat_t mat);
void dmat_init(dmat_t mat, int r, int c);
void dmat_clear(dmat_t mat);
void dmat_get_arb_mat(arb_mat_t dst, const dmat_t src);


typedef struct
{
    csr_graph_t g;
    dmat_t mat;
    pmat_t p;
    int root_node_index;
    int *preorder;
    int use_equilibrium_root_prior;
    double rate_divisor;
    double *edge_rate_coefficients;
    csr_edge_mapper_t edge_map;
} model_and_data_struct;
typedef model_and_data_struct model_and_data_t[1];

void model_and_data_init(model_and_data_t m);
void model_and_data_clear(model_and_data_t m);


#ifdef __cplusplus
}
#endif

#endif
