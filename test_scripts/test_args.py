"""
Check what happens when bad arguments are passed the the programs.

"""
from __future__ import print_function, division

import os
import json
import copy
from subprocess import Popen, PIPE
from numpy.testing import assert_equal, assert_raises, TestCase

args = ['arbplf-ll']

class ReturnError(Exception):
    pass

good_input = {
     "model_and_data" : {
         "edges" : [[0, 1], [1, 2], [1, 3]],
         "edge_rate_coefficients" : [2.0, 4.2, 0.5],
         "rate_matrix" : [
             [0, 4.2, 3.0],
             [1.0, 0, 5.0],
             [6.0, 0.5, 0]],
         "probability_array" : [
             [[0.6, 0.2, 0.2], [1, 1, 1], [1, 0, 0], [0, 0, 1]],
             [[0.6, 0.2, 0.2], [1, 1, 1], [0, 0, 1], [0, 0, 1]]]},
     "reductions" : [{
         "columns" : ["site"],
         "aggregation" : "sum"}],
     "working_precision" : 256,
     "sum_product_strategy" : "brute_force"}

def runjson(args, d):
    s_in = json.dumps(d)
    p = Popen(args, stdout=PIPE, stdin=PIPE, stderr=PIPE)
    data = p.communicate(input=s_in)
    out, err = data
    print('out:', out)
    print('err:', err)
    if p.returncode:
        raise ReturnError(err)
    else:
        return json.loads(out)

def test_ok():
    runjson(args, good_input)

def test_bad_edge_not_array_but_dict():
    x = copy.deepcopy(good_input)
    x['model_and_data']['edges'][2] = {'hello' : 'world'}
    assert_raises(ReturnError, runjson, args, x)

def test_bad_edge_not_array_but_string():
    x = copy.deepcopy(good_input)
    x['model_and_data']['edges'][2] = 'hello'
    assert_raises(ReturnError, runjson, args, x)

def test_bad_edge_not_array_but_int():
    x = copy.deepcopy(good_input)
    x['model_and_data']['edges'][2] = 0
    assert_raises(ReturnError, runjson, args, x)

def test_bad_edge_negative_node_index():
    x = copy.deepcopy(good_input)
    x['model_and_data']['edges'][2] = [-1, 3]
    assert_raises(ReturnError, runjson, args, x)

def test_bad_edge_too_high_node_index():
    x = copy.deepcopy(good_input)
    x['model_and_data']['edges'] = [[0, 1], [1, 5], [1, 6]]
    assert_raises(ReturnError, runjson, args, x)

def test_bad_edge_long_array():
    x = copy.deepcopy(good_input)
    x['model_and_data']['edges'][0] = [0, 1, 2]
    assert_raises(ReturnError, runjson, args, x)

def test_bad_edge_short_array():
    x = copy.deepcopy(good_input)
    x['model_and_data']['edges'][0] = [0]
    assert_raises(ReturnError, runjson, args, x)

def test_bad_edge_connected_loop():
    x = copy.deepcopy(good_input)
    x['model_and_data']['edges'][2] = [[0, 1], [1, 2], [2, 2]]
    assert_raises(ReturnError, runjson, args, x)

def test_bad_edge_disconnected_loop():
    x = copy.deepcopy(good_input)
    x['model_and_data']['edges'][2] = [[0, 1], [1, 2], [3, 3]]
    assert_raises(ReturnError, runjson, args, x)

def test_bad_edges_dag_but_not_tree():
    x = copy.deepcopy(good_input)
    x['model_and_data']['edges'] = [[0, 1], [1, 2], [1, 3], [2, 3]]
    assert_raises(ReturnError, runjson, args, x)

def test_bad_edges_cycle():
    x = copy.deepcopy(good_input)
    x['model_and_data']['edges'] = [[0, 1], [1, 2], [2, 0]]
    assert_raises(ReturnError, runjson, args, x)

def test_bad_edges_disconnected_dag_and_tree():
    # This graph shares some properties with trees, but is not a tree.
    # It is a directed acyclic graph, and the number of nodes in the graph
    # is one more than the number of edges.
    # It is bipartite.
    x = copy.deepcopy(good_input)
    x['model_and_data']['edges'] = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5]]
    assert_raises(ReturnError, runjson, args, x)
