#include "csr_graph.h"

void
csr_edge_mapper_pre_init(csr_edge_mapper_t g)
{
    g->n = 0;
    g->accum_nnz = 0;
    g->accum_out_degree = NULL;
    g->order = NULL;
}

void
csr_edge_mapper_init(csr_edge_mapper_t g, int node_count, int edge_count)
{
    g->n = node_count;
    g->accum_nnz = 0;
    g->accum_out_degree = calloc(node_count, sizeof(int));
    g->order = calloc(edge_count, sizeof(int));
}

void
csr_edge_mapper_clear(csr_edge_mapper_t g)
{
    free(g->order);
    free(g->accum_out_degree);
}

void
csr_edge_mapper_add_edge(csr_edge_mapper_t m, csr_graph_t g, int a, int b)
{
    int idx;
    idx = g->indptr[a] + m->accum_out_degree[a];
    if (idx >= g->indptr[a+1])
    {
        fprintf(stderr, "internal error (indptr) in edge mapping\n");
        abort();
    }
    if (g->indices[idx] != b)
    {
        fprintf(stderr, "internal error (indices) in edge mapping\n");
        abort();
    }
    m->order[m->accum_nnz] = idx;
    m->accum_nnz++;
    m->accum_out_degree[a]++;
}





void
csr_graph_init(csr_graph_t g)
{
    g->indices = NULL;
    g->indptr = NULL;
    g->n = 0;
    g->nnz = 0;
    g->_edge_builder = NULL;
}

void
csr_graph_init_outdegree(csr_graph_t g, int n, int *out_degree)
{
    int i, nnz, accum;

    /* count the edges, and use this count to allocate the indices array  */
    accum = 0;
    for (i = 0; i < n; i++)
    {
        accum += out_degree[i];
    }
    nnz = accum;

    /* define the indptr array */
    accum = 0;
    g->indptr = malloc((n+1) * sizeof(int));
    for (i = 0; i < n; i++)
    {
        g->indptr[i] = accum;
        accum += out_degree[i];
    }
    g->indptr[i] = accum;

    g->n = n;
    g->nnz = nnz;
}

void
csr_graph_clear(csr_graph_t g)
{
    free(g->indices);
    free(g->indptr);
    free(g->_edge_builder);
    g->indices = NULL;
    g->indptr = NULL;
    g->_edge_builder = NULL;
    g->n = 0;
    g->nnz = 0;
}

int
csr_graph_get_tree_topo_sort(
        int *preorder, const csr_graph_t g, int root_node_index)
{
    /* the graph must be a directed tree with a known root */
    int i;
    int nu, nv;
    int *visited, *u, *v;
    int *tmp;
    int npre;
    int result;

    result = 0;
    visited = NULL;
    u = NULL;
    v = NULL;

    if (!(0 <= root_node_index && root_node_index < g->n))
    {
        fprintf(stderr, "validate_edges: invalid root node index: %d\n",
                root_node_index);
        result = -1;
        goto finish;
    }

    visited = calloc(g->n, sizeof(int));
    u = malloc(g->n * sizeof(int));
    v = malloc(g->n * sizeof(int));
    npre = 0;
    nv = 1;
    v[0] = root_node_index;
    visited[root_node_index] = 1;
    while (nv)
    {
        int a, b, j;
        tmp = u; u = v; v = tmp;
        nu = nv;
        nv = 0;
        for (i = 0; i < nu; i++)
        {
            a = u[i];
            preorder[npre++] = a;
            for (j = g->indptr[a]; j < g->indptr[a+1]; j++)
            {
                b = g->indices[j];
                if (visited[b])
                {
                    fprintf(stderr, "validate_edges: topo sort failed: ");
                    fprintf(stderr, "node index %d already visited\n", b);
                    result = -1;
                    goto finish;
                }
                else
                {
                    v[nv++] = b;
                    visited[b] = 1;
                }
            }
        }
    }
    if (npre != g->n)
    {
        fprintf(stderr, "validate_edges: the topo sort contains ");
        fprintf(stderr, "%d of the %d nodes\n", npre, g->n);
        result = -1;
        goto finish;
    }

finish:

    free(u);
    free(v);
    free(visited);

    return result;
}

void
csr_graph_start_adding_edges(csr_graph_t g)
{
    int i;
    if (!g->indptr || g->indices || g->_edge_builder)
    {
        fprintf(stderr, "tree construction error\n");
        abort();
    }
    g->_edge_builder = calloc(g->n, sizeof(int));
    g->indices = malloc(g->nnz * sizeof(int));
    for (i = 0; i < g->nnz; i++)
    {
        g->indices[i] = -1;
    }
}

void
csr_graph_stop_adding_edges(csr_graph_t g)
{
    int i;
    if (!g->_edge_builder)
    {
        fprintf(stderr, "tree construction error\n");
        abort();
    }
    for (i = 0; i < g->nnz; i++)
    {
        if (g->indices[i] < 0)
        {
            fprintf(stderr, "prematurely stopped adding edges\n");
            abort();
        }
    }
    free(g->_edge_builder);
    g->_edge_builder = NULL;
}

void
csr_graph_add_edge(csr_graph_t g, int a, int b)
{
    int offset;
    if (!g->_edge_builder)
    {
        fprintf(stderr, "tree construction error\n");
        abort();
    }
    offset = g->indptr[a] + g->_edge_builder[a];
    g->indices[offset] = b;
    g->_edge_builder[a]++;
}
