from __future__ import print_function, division

import json
import copy
from numpy.testing import assert_equal, assert_raises, TestCase

from arbplf import arbplf_ll

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
     "site_reduction" : {
         "aggregation" : "sum"},
     }

def _myfail(d):
    s = json.dumps(d)
    assert_raises(RuntimeError, arbplf_ll, s)

def test_ok():
    arbplf_ll(json.dumps(good_input))

def test_bad_edge_not_array_but_dict():
    x = copy.deepcopy(good_input)
    x['model_and_data']['edges'][2] = {'hello' : 'world'}
    yield _myfail, x

def test_bad_edge_not_array_but_string():
    x = copy.deepcopy(good_input)
    x['model_and_data']['edges'][2] = 'hello'
    yield _myfail, x

def test_bad_edge_not_array_but_int():
    x = copy.deepcopy(good_input)
    x['model_and_data']['edges'][2] = 0
    yield _myfail, x

def test_bad_edge_negative_node_index():
    x = copy.deepcopy(good_input)
    x['model_and_data']['edges'][2] = [-1, 3]
    yield _myfail, x

def test_bad_edge_too_high_node_index():
    x = copy.deepcopy(good_input)
    x['model_and_data']['edges'] = [[0, 1], [1, 5], [1, 6]]
    yield _myfail, x

def test_bad_edge_long_array():
    x = copy.deepcopy(good_input)
    x['model_and_data']['edges'][0] = [0, 1, 2]
    yield _myfail, x

def test_bad_edge_short_array():
    x = copy.deepcopy(good_input)
    x['model_and_data']['edges'][0] = [0]
    yield _myfail, x

def test_bad_edge_connected_loop():
    x = copy.deepcopy(good_input)
    x['model_and_data']['edges'] = [[0, 1], [1, 2], [2, 2]]
    yield _myfail, x

def test_bad_edge_disconnected_loop():
    x = copy.deepcopy(good_input)
    x['model_and_data']['edges'] = [[0, 1], [1, 2], [3, 3]]
    yield _myfail, x

def test_bad_edges_dag_but_not_tree():
    x = copy.deepcopy(good_input)
    x['model_and_data']['edges'] = [[0, 1], [1, 2], [1, 3], [2, 3]]
    yield _myfail, x

def test_bad_edges_cycle():
    x = copy.deepcopy(good_input)
    x['model_and_data']['edges'] = [[0, 1], [1, 2], [2, 0]]
    yield _myfail, x

def test_bad_edges_disconnected_cycle():
    # This graph looks like a tree locally and using global degree statistics,
    # but it is disconnected and has a cycle.
    x = copy.deepcopy(good_input)
    x['model_and_data']['edges'] = [
            [0, 1], [1, 2], [2, 0],
            [3, 4], [4, 5], [4, 6]]
    yield _myfail, x

def test_bad_edges_undirected_tree():
    # This graph is an undirected tree but not a directed tree.
    x = copy.deepcopy(good_input)
    x['model_and_data']['edges'] = [[0, 1], [2, 1], [1, 3]]
    yield _myfail, x

def test_bad_edges_disconnected_dag_and_tree():
    # This graph shares some properties with trees, but is not a tree.
    # It is a directed acyclic graph, and the number of nodes in the graph
    # is one more than the number of edges.
    # It is bipartite.
    x = copy.deepcopy(good_input)
    x['model_and_data']['edges'] = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5]]
    yield _myfail, x

def test_bad_edge_coeffs_not_array():
    x = copy.deepcopy(good_input)
    m = x['model_and_data']
    m['edge_rate_coefficients'] = {"hello" : "world"}
    yield _myfail, x

def test_bad_edge_coeffs_too_many():
    x = copy.deepcopy(good_input)
    m = x['model_and_data']
    m['edge_rate_coefficients'] = [1, 2, 3, 4]
    yield _myfail, x

def test_bad_edge_coeffs_too_few():
    x = copy.deepcopy(good_input)
    m = x['model_and_data']
    m['edge_rate_coefficients'] = [1, 2]
    yield _myfail, x

def test_bad_edge_coeffs_string():
    x = copy.deepcopy(good_input)
    m = x['model_and_data']
    m['edge_rate_coefficients'] = [1, 2, "wat"]
    yield _myfail, x

def test_bad_edge_coeffs_number_as_string():
    x = copy.deepcopy(good_input)
    m = x['model_and_data']
    m['edge_rate_coefficients'] = [1, 2, "3"]
    yield _myfail, x

def test_bad_edge_coeffs_negative():
    x = copy.deepcopy(good_input)
    m = x['model_and_data']
    m['edge_rate_coefficients'] = [-1, 2, 3]
    yield _myfail, x

def test_bad_rate_matrix_not_array():
    x = copy.deepcopy(good_input)
    m = x['model_and_data']['rate_matrix'] = {"hello" : "world"}
    yield _myfail, x

def test_bad_rate_matrix_row_is_not_array():
    x = copy.deepcopy(good_input)
    m = x['model_and_data']['rate_matrix'] = [
            [0, 4.2, 3.0],
            {"hello" : "world"},
            [6.0, 0.5, 0]]
    yield _myfail, x

def test_bad_rate_matrix_row_is_too_long():
    x = copy.deepcopy(good_input)
    m = x['model_and_data']['rate_matrix'] = [
            [0, 4.2, 3.0],
            [1.0, 0, 5.0, 8.0],
            [6.0, 0.5, 0]]
    yield _myfail, x

def test_bad_rate_matrix_row_is_too_short():
    x = copy.deepcopy(good_input)
    m = x['model_and_data']['rate_matrix'] = [
            [0, 4.2, 3.0],
            [1.0, 0],
            [6.0, 0.5, 0]]
    yield _myfail, x

def test_bad_rate_matrix_string_entry():
    x = copy.deepcopy(good_input)
    m = x['model_and_data']['rate_matrix'] = [
            [0, 4.2, 3.0],
            [1.0, "wat", 5.0],
            [6.0, 0.5, 0]]
    yield _myfail, x

def test_bad_rate_matrix_number_as_string_entry():
    x = copy.deepcopy(good_input)
    m = x['model_and_data']['rate_matrix'] = [
            [0, 4.2, 3.0],
            [1.0, "42", 5.0],
            [6.0, 0.5, 0]]
    yield _myfail, x

def test_bad_rate_matrix_negative_entry():
    x = copy.deepcopy(good_input)
    m = x['model_and_data']['rate_matrix'] = [
            [0, 4.2, 3.0],
            [1.0, 0, -5.0],
            [6.0, 0.5, 0]]
    yield _myfail, x

def test_bad_probability_array_bad_data_level_0():
    x = copy.deepcopy(good_input)
    m = x['model_and_data']['probability_array'] = "foo"
    yield _myfail, x

def test_bad_probability_array_bad_data_level_1a():
    x = copy.deepcopy(good_input)
    m = x['model_and_data']['probability_array'][0] = "foo"
    yield _myfail, x

def test_bad_probability_array_bad_data_level_1b():
    x = copy.deepcopy(good_input)
    m = x['model_and_data']['probability_array'][1] = "foo"
    yield _myfail, x

def test_bad_probability_array_bad_data_level_2a():
    x = copy.deepcopy(good_input)
    m = x['model_and_data']['probability_array'][0][0] = "foo"
    yield _myfail, x

def test_bad_probability_array_bad_data_level_2b():
    x = copy.deepcopy(good_input)
    m = x['model_and_data']['probability_array'][1][1] = "foo"
    yield _myfail, x

def test_bad_probability_array_bad_data_level_3a():
    x = copy.deepcopy(good_input)
    m = x['model_and_data']['probability_array'][0][0][0] = "foo"
    yield _myfail, x

def test_bad_probability_array_bad_data_level_3b():
    x = copy.deepcopy(good_input)
    m = x['model_and_data']['probability_array'][1][1][1] = "foo"
    yield _myfail, x

def test_bad_rate_matrix_too_many_nodes():
    x = copy.deepcopy(good_input)
    m = x['model_and_data']['probability_array'] = [
             [[0.6, 0.2, 0.2], [1,1,1], [1,0,0], [0,0,1], [1,1,1]],
             [[0.6, 0.2, 0.2], [1,1,1], [0,0,1], [0,0,1], [1,1,1]]]
    yield _myfail, x

def test_bad_rate_matrix_too_few_nodes():
    x = copy.deepcopy(good_input)
    m = x['model_and_data']['probability_array'] = [
             [[0.6, 0.2, 0.2], [0,0,1], [1,1,1]],
             [[0.6, 0.2, 0.2], [0,0,1], [1,1,1]]]
    yield _myfail, x

def test_bad_rate_matrix_too_many_states():
    x = copy.deepcopy(good_input)
    m = x['model_and_data']['probability_array'] = [
             [[0.6, 0.2, 0.2, 0.1], [1,1,1,1], [1,0,0,1], [0,0,1,1]],
             [[0.6, 0.2, 0.2, 0.1], [1,1,1,1], [0,0,1,1], [0,0,1,1]]]
    yield _myfail, x

def test_bad_rate_matrix_too_few_states():
    x = copy.deepcopy(good_input)
    m = x['model_and_data']['probability_array'] = [
             [[0.6, 0.2], [1,1], [1,0], [0,0]],
             [[0.6, 0.2], [1,1], [0,0], [0,0]]]
    yield _myfail, x
