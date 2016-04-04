"""
Test a path model.
In particular, test that edges are indexed correctly.

Directed path tree, with corresponding states:
 0 --> 1 --> 2 --> 3 --> 4
(0)   (1)   (1)   (2)   (2)

Directed rate matrix:
0 --> 1 --> 2

The number and type of transition on each edge is completely known,
due to the observed data at the nodes and due to the nature
of the rate matrix.

"""
from __future__ import print_function, division

from StringIO import StringIO
import json
import copy
import random

import numpy as np
from numpy.testing import (
        assert_, assert_equal, assert_raises, assert_allclose, TestCase)

from arbplf import arbplf_ll
from arbplf import arbplf_dwell
from arbplf import arbplf_em_update

def _shuffle_edges(d_in):
    """
    This should affect edge outputs according to a returned permutation.
    adjust edges
    adjust edge rate coefficients
    """
    d = copy.deepcopy(d_in)
    m = d['model_and_data']
    edge_count = len(m['edges'])
    node_count = edge_count + 1
    perm = range(edge_count)
    random.shuffle(perm)
    # reorder edges
    edges_in = m['edges']
    next_edges = [[] for i in range(edge_count)]
    for a in range(edge_count):
        next_edges[perm[a]] = edges_in[a][:]
    d['model_and_data']['edges'] = next_edges
    # reorder edge rate coefficients
    p = d['model_and_data']['edge_rate_coefficients']
    p_in = d_in['model_and_data']['edge_rate_coefficients']
    for a in range(edge_count):
        p[perm[a]] = p_in[a]
    # return the shuffled input
    print(perm)
    print(d_in)
    print(d)
    print()
    return d, perm

def _perm_output_edges(d, perm):
    edge_count = len(perm)
    invperm = [None]*edge_count
    for i, p in enumerate(perm):
        invperm[p] = i
    # get the edge column index
    edge_column_idx = None
    for i, name in enumerate(d['columns']):
        if name == 'edge':
            edge_column_idx = i
    # permute the edge indices
    d_out = copy.deepcopy(d)
    row_count = len(d['data'])
    for row_idx in range(row_count):
        original_edge_idx = d['data'][row_idx][edge_column_idx]
        new_edge_idx = invperm[original_edge_idx]
        #new_edge_idx = perm[original_edge_idx]
        d_out['data'][row_idx][edge_column_idx] = new_edge_idx
    # return the analysis with permuted edge labels
    return d_out


def _shuffle_nodes(d_in):
    """
    This should not affect outputs that are indexed by site, edge, and state.
    """
    d = copy.deepcopy(d_in)
    m = d['model_and_data']
    edge_count = len(m['edges'])
    node_count = edge_count + 1
    perm = range(node_count)
    random.shuffle(perm)
    # update edges according to the relabeled nodes
    next_edges = []
    for a, b in m['edges']:
        next_edges.append([perm[a], perm[b]])
    d['model_and_data']['edges'] = next_edges
    # update the probability array according to the relabeled nodes
    next_array = []
    for site_array_in in m['probability_array']:
        site_array_out = [[] for i in range(node_count)]
        for a in range(node_count):
            site_array_out[perm[a]] = site_array_in[a][:]
        next_array.append(site_array_out)
    d['model_and_data']['probability_array'] = next_array
    # return the shuffled input
    return d


d_original = {
            "model_and_data" : {
                "edges" : [
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 4]],
                "edge_rate_coefficients" : [1, 1, 2, 3],
                "rate_matrix" : [
                    [0, 1, 0],
                    [0, 0, 1],
                    [0, 0, 0]],
                "probability_array" : [[
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [0, 0, 1]]]
                }
            }


def test_shuffled_nodes():

    d = copy.deepcopy(d_original)
    original_dwell = json.loads(arbplf_dwell(json.dumps(d)))

    d = copy.deepcopy(d_original)
    original_ll = json.loads(arbplf_ll(json.dumps(d)))

    d = copy.deepcopy(d_original)
    d['site_reduction'] = {'aggregation' : 'sum'}
    original_em_update = json.loads(arbplf_em_update(json.dumps(d)))

    iter_count = 10
    for i in range(iter_count):
        d_shuffled = _shuffle_nodes(d_original)

        d = copy.deepcopy(d_shuffled)
        dwell = json.loads(arbplf_dwell(json.dumps(d)))
        assert_equal(dwell, original_dwell)

        d = copy.deepcopy(d_shuffled)
        ll = json.loads(arbplf_ll(json.dumps(d)))
        assert_equal(ll, original_ll)

        d = copy.deepcopy(d_shuffled)
        d['site_reduction'] = {'aggregation' : 'sum'}
        em_update = json.loads(arbplf_em_update(json.dumps(d)))
        assert_equal(em_update, original_em_update)


def test_shuffled_edges():

    d = copy.deepcopy(d_original)
    original_dwell = json.loads(arbplf_dwell(json.dumps(d)))

    d = copy.deepcopy(d_original)
    original_ll = json.loads(arbplf_ll(json.dumps(d)))

    d = copy.deepcopy(d_original)
    d['site_reduction'] = {'aggregation' : 'sum'}
    original_em_update = json.loads(arbplf_em_update(json.dumps(d)))

    iter_count = 10
    for i in range(iter_count):
        d_shuffled, perm = _shuffle_edges(d_original)

        # the ll output does not have an edge column
        d = copy.deepcopy(d_shuffled)
        ll = json.loads(arbplf_ll(json.dumps(d)))
        assert_equal(ll, original_ll)

        d = copy.deepcopy(d_shuffled)
        dwell = json.loads(arbplf_dwell(json.dumps(d)))
        dwell_prime = _perm_output_edges(dwell, perm)
        dwell_prime['data'].sort()
        assert_equal(dwell_prime, original_dwell)

        d = copy.deepcopy(d_shuffled)
        d['site_reduction'] = {'aggregation' : 'sum'}
        em_update = json.loads(arbplf_em_update(json.dumps(d)))
        em_update_prime = _perm_output_edges(em_update, perm)
        em_update_prime['data'].sort()
        assert_equal(em_update_prime, original_em_update)
