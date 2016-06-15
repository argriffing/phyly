"""
Compare to GeLL and to PAML log likelihood.

This test does not search for a maximum likelihood.
GeLL: /test/Likelihood/LikelihoodTest.java

nucleotide order is TCAG.

"""
from __future__ import print_function, division

import json

import dendropy

# define a node order
nodes = ['Human', 'Chimpanzee', 'Gorilla', 'Orangutan', 'Gibbon', 'A', 'B', 'C']

newick = """
(((Human: 0.057987, Chimpanzee: 0.074612)A: 0.035490,
Gorilla: 0.074352)B: 0.131394, Orangutan: 0.350156, Gibbon: 0.544601)C;
"""

root_prior = [0.23500, 0.34587, 0.32300, 0.09613]

rate_matrix = [
        [0, 1.370596, 0.039081, 0.000004],
        [0.931256, 0, 0.072745, 0.004875],
        [0.028434, 0.077896, 0, 0.439244],
        [0.000011, 0.017541, 1.475874, 0]]

gamma_shape = 0.19249

t = dendropy.Tree.get(data=newick, schema='newick')

def nodestr(node):
    # dendropy cares way too much about the distinction between internal
    # nodes and leaf nodes...
    if node.label is not None:
        return node.label
    if node.taxon is not None:
        return node.taxon.label

edges = []
edge_rate_coefficients = []
for edge in t.edges():
    tail = edge.tail_node
    head = edge.head_node
    blen = edge.length
    if tail is not None:
        a = nodes.index(nodestr(tail))
        b = nodes.index(nodestr(head))
        edges.append([a, b])
        edge_rate_coefficients.append(blen)

d = dict(model_and_data = dict(
    edges = edges,
    edge_rate_coefficients = edge_rate_coefficients,
    gamma_rate_mixture = dict(gamma_shape=gamma_shape, gamma_categories=4),
    root_prior = root_prior,
    rate_matrix = rate_matrix,
    ))
s = json.dumps(d, indent=2)
print(s)
