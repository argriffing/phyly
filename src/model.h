#ifndef MODEL_H
#define MODEL_H

#include "arb.h"
#include "arb_mat.h"

#include "csr_graph.h"
#include "rate_mixture.h"


#ifdef __cplusplus
extern "C" {
#endif


enum root_prior_mode {
    ROOT_PRIOR_UNDEFINED,
    ROOT_PRIOR_NONE,
    ROOT_PRIOR_CUSTOM,
    ROOT_PRIOR_UNIFORM,
    ROOT_PRIOR_EQUILIBRIUM};

typedef struct
{
    int n;
    enum root_prior_mode mode;
    double *custom_distribution;
} root_prior_struct;
typedef root_prior_struct root_prior_t[1];

void root_prior_pre_init(root_prior_t r);
void root_prior_init(root_prior_t r, int n, enum root_prior_mode mode);
void root_prior_clear(root_prior_t r);
void root_prior_mul_col_vec(arb_mat_t A, const root_prior_t r,
        const arb_struct *equilibrium, slong prec);
void root_prior_expectation(arb_t out,
        const root_prior_t r, const arb_mat_t A,
        const arb_struct *equilibrium, slong prec);


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
        arb_mat_struct *base_node_vectors, const pmat_t p, slong site);


typedef struct
{
    int *preorder;
    int *b_to_idx;
    int *idx_to_a;
} navigation_struct;
typedef navigation_struct navigation_t[1];

void navigation_pre_init(navigation_t navigation);
int navigation_init(navigation_t nav, const csr_graph_t g,
        slong node_count, slong edge_count, slong root_node_index);
void navigation_clear(navigation_t navigation);


typedef struct
{
    csr_graph_t g;
    arb_mat_t mat;
    pmat_t p;
    int root_node_index;
    int use_equilibrium_rate_divisor;
    arb_t rate_divisor;
    root_prior_t root_prior;
    double *edge_rate_coefficients;
    csr_edge_mapper_t edge_map;
    rate_mixture_t rate_mixture;
    navigation_t navigation;
} model_and_data_struct;
typedef model_and_data_struct model_and_data_t[1];

void model_and_data_init(model_and_data_t m);
void model_and_data_clear(model_and_data_t m);
slong model_and_data_edge_count(const model_and_data_t m);
slong model_and_data_node_count(const model_and_data_t m);
slong model_and_data_state_count(const model_and_data_t m);
slong model_and_data_site_count(const model_and_data_t m);
slong model_and_data_rate_category_count(const model_and_data_t m);
int model_and_data_uses_equilibrium(const model_and_data_t m);




/* Update using a new level of precision. */
void
_update_rate_matrix_and_equilibrium(
        arb_mat_t rate_matrix,
        arb_struct *equilibrium,
        arb_t rate_divisor,
        int use_equilibrium_rate_divisor,
        const root_prior_t root_prior,
        const rate_mixture_t rate_mixture,
        const arb_mat_t mat,
        slong prec);

#ifdef __cplusplus
}
#endif

#endif
