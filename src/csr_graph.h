#ifndef CSR_GRAPH_H
#define CSR_GRAPH_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>


#ifdef __cplusplus
extern "C" {
#endif


typedef struct
{
    int *indices;
    int *indptr;
    int n;
    int nnz;
    int *_edge_builder;
} csr_graph_struct;
typedef csr_graph_struct csr_graph_t[1];

void csr_graph_init(csr_graph_t g);
void csr_graph_init_outdegree(csr_graph_t g, int n, int *out_degree);
void csr_graph_clear(csr_graph_t g);
void csr_graph_start_adding_edges(csr_graph_t g);
void csr_graph_stop_adding_edges(csr_graph_t g);
void csr_graph_add_edge(csr_graph_t g, int a, int b);
int csr_graph_get_tree_topo_sort(
        int *preorder, const csr_graph_t g, int root_node_index);


#ifdef __cplusplus
}
#endif

#endif
