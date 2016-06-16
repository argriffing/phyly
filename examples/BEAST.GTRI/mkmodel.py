"""
https://github.com/beast-dev/beast-mcmc/blob/master/src/test/dr/evomodel/treelikelihood/LikelihoodTest.java

The model is GTR with empirical nucleotide frequencies,
and the nucleotide order is TCAG.
Half of the sites are assumed to be invariable and the other
half are assumed to evolve at double speed.

combine with nucleotide data:
$ python mknuc.py > nuc.json
$ python mkmodel.py > model.json
$ jq -s '.[0] * .[1]' nuc.json model.json > in.json

This script is mostly copied and pasted from the GTR example.

"""
from __future__ import print_function, division

import itertools
import json

import dendropy

# define a node order

nodes = [
    'human',
    'chimp',
    'bonobo',
    'N3',
    'N4',
    'gorilla',
    'N6',
    'orangutan',
    'N8',
    'siamang',
    'R']

newick = "((((human:0.024003,(chimp:0.010772,bonobo:0.010772)N3:0.013231)N4:0.012035, gorilla:0.036038)N6:0.033087,orangutan:0.069125)N8:0.030457,siamang:0.099582)R;"

root_prior = [
        0.26121198857732775,
        0.24655436448599188,
        0.33909428971935046,
        0.1531393572173299]

rate_matrix = [
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0]]

# This is how BEAST GTR modifies a symmetric matrix
# to control the stationary distribution.
for i, j in itertools.permutations((0, 1, 2, 3), 2):
    rate_matrix[i][j] *= root_prior[j]

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
    rate_mixture = dict(rates=[0, 2], prior=[0.5, 0.5]),
    rate_divisor = "equilibrium_exit_rate",
    rate_matrix = rate_matrix,
    ))
s = json.dumps(d, indent=2)
print(s)
