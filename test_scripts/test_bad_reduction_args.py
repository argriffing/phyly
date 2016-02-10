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

def test_reduction_bad_value():
    x = copy.deepcopy(good_input)
    x['site_reduction']['hello'] = 'world'
    yield _myfail, x

def test_aggregation_bad_value_dict():
    x = copy.deepcopy(good_input)
    x['site_reduction']['aggregation'] = {"hello" : "world"}
    yield _myfail, x

def test_aggregation_bad_value_string():
    x = copy.deepcopy(good_input)
    x['site_reduction']['aggregation'] = "foo"
    yield _myfail, x

def test_aggregation_bad_weight_string():
    x = copy.deepcopy(good_input)
    x['site_reduction']['aggregation'] = ["42", "3"]
    yield _myfail, x

def test_aggregation_too_few_weights():
    x = copy.deepcopy(good_input)
    x['site_reduction']['aggregation'] = [42.0]
    yield _myfail, x

def test_aggregation_too_many_weights():
    x = copy.deepcopy(good_input)
    x['site_reduction']['aggregation'] = [42.0, 43.0, 44.0]
    yield _myfail, x

def test_selection_bad_value_dict():
    x = copy.deepcopy(good_input)
    x['site_reduction']['selection'] = {"hello" : "world"}
    yield _myfail, x

def test_selection_bad_value_string():
    x = copy.deepcopy(good_input)
    x['site_reduction']['selection'] = "hello"
    yield _myfail, x

def test_selection_negative_index():
    x = copy.deepcopy(good_input)
    x['site_reduction']['selection'] = [0, -2]
    yield _myfail, x

def test_selection_too_large_index():
    x = copy.deepcopy(good_input)
    x['site_reduction']['selection'] = [100, 0]
    yield _myfail, x

def test_selection_bad_index_float():
    x = copy.deepcopy(good_input)
    x['site_reduction']['selection'] = [3.14]
    yield _myfail, x

def test_selection_bad_index_string():
    x = copy.deepcopy(good_input)
    x['site_reduction']['selection'] = ["0"]
    yield _myfail, x

def test_selection_bad_index_dict():
    x = copy.deepcopy(good_input)
    x['site_reduction']['selection'] = [{"hello" : "world"}]
    yield _myfail, x

def test_selection_aggregation_too_few_weights_2():
    x = copy.deepcopy(good_input)
    x['site_reduction']['selection'] = [0, 1, 0, 1]
    x['site_reduction']['aggregation'] = [0.1, 0.2]
    yield _myfail, x

def test_selection_aggregation_too_few_weights_3():
    x = copy.deepcopy(good_input)
    x['site_reduction']['selection'] = [0, 1, 0, 1]
    x['site_reduction']['aggregation'] = [0.1, 0.2, 0.3]
    yield _myfail, x

def test_selection_aggregation_too_many_weights():
    x = copy.deepcopy(good_input)
    x['site_reduction']['selection'] = [0, 1, 0, 1]
    x['site_reduction']['aggregation'] = [0.1, 0.2, 0.3, 0.4, 0.5]
    yield _myfail, x
