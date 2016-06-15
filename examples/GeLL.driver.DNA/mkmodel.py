"""
Test log likelihood under a DNA model with discretized gamma rates at sites.

Compare to the GeLL DNA example available at
http://phylo.bio.ku.edu/GeLL/DriverExamples.tar.gz

nucleotide order is TCAG

brown.nuc:
human chimpanzee gorilla orangutan gibbon
0     1          2       3         4
"""
from __future__ import print_function, division

from StringIO import StringIO
import json

import scipy
import scipy.stats
from scipy.special import gammainc
import numpy as np

import dendropy

# Impose a node order.
nodes = ['Human', 'Chimpanzee', 'Gorilla', 'Orangutan', 'Gibbon', 'A', 'B', 'C']

newick = '((Gorilla:0.07435276303709204, (Chimpanzee:0.07462376514241155,Human:0.057990312243662766)A:0.0355060511914654)B:0.13146876021580028, Gibbon:0.5448816311811152, Orangutan:0.350348976027984)C;'

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

gamma_shape = 0.19242344607262146
rate_category_count = 4

pT = 1.0
pC = 1.471797772941626
pA = 1.3744425342086881
pG = 0.40904476825485114
root_prior = np.array([pT, pC, pA, pG])
root_prior = root_prior / root_prior.sum()

a = 0.8671005148903497
b = 0.026470048816886478
c = 0.0
d = 0.049264783448354395
e = 0.011075451195349906
rate_matrix = np.array([
    [0, a, b, c],
    [a, 0, d, e],
    [b, d, 0, 1],
    [c, e, 1, 0]])
rate_matrix = rate_matrix * root_prior

exit_rates = rate_matrix.sum(axis=1)
expected_rate = np.dot(root_prior, exit_rates)
rate_matrix = rate_matrix / expected_rate

rate_matrix = rate_matrix.tolist()
root_prior = root_prior.tolist()

d = dict(model_and_data = dict(
    edges = edges,
    edge_rate_coefficients = edge_rate_coefficients,
    gamma_rate_mixture = dict(gamma_shape=gamma_shape, gamma_categories=4),
    rate_divisor = "equilibrium_exit_rate",
    root_prior = root_prior,
    rate_matrix = rate_matrix,
    ))
s = json.dumps(d, indent=2)
print(s)
