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


/*
order[i] gives the csr index of the i'th added edge.
The purpose of this map is to allow data to be associated with
edges of the graph, after the graph has been built.
The csr indptr array is assumed to already exist.
accum_out_degree[i] gives the current number of out degrees of node i
*/
typedef struct
{
    int n;
    int accum_nnz;
    int *order;
    int *accum_out_degree;
} csr_edge_mapper_struct;
typedef csr_edge_mapper_struct csr_edge_mapper_t[1];

void csr_edge_mapper_pre_init(csr_edge_mapper_t g);
void csr_edge_mapper_init(csr_edge_mapper_t g, int node_count, int edge_count);
void csr_edge_mapper_clear(csr_edge_mapper_t g);
void csr_edge_mapper_add_edge(csr_edge_mapper_t m, csr_graph_t g, int a, int b);


#ifdef __cplusplus
}
#endif

#endif
