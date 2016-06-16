"""
https://github.com/beast-dev/beast-mcmc/blob/master/src/test/dr/evomodel/treelikelihood/LikelihoodTest.java

nucleotide order is TCAG.

use jukes cantor

combine with nucleotide data:
$ python mknuc.py > nuc.json
$ python mkmodel.py > model.json
$ jq -s '.[0] * .[1]' nuc.json model.json > in.json

"""
from __future__ import print_function, division

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

root_prior = [0.25, 0.25, 0.25, 0.25]

rate_matrix = [
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0]]

rate_divisor = 3

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
    root_prior = root_prior,
    rate_divisor = "equilibrium_exit_rate",
    rate_matrix = rate_matrix,
    ))
s = json.dumps(d, indent=2)
print(s)
