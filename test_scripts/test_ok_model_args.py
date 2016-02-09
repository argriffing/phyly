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

def test_ok():
    arbplf_ll(json.dumps(good_input))

def test_ok_reduction_deleted():
    x = copy.deepcopy(good_input)
    del x['site_reduction']
    arbplf_ll(json.dumps(x))

def test_ok_reduction_empty():
    x = copy.deepcopy(good_input)
    x['site_reduction'] = {}
    arbplf_ll(json.dumps(x))

def test_ok_reduction_avg_aggregation():
    x = copy.deepcopy(good_input)
    x['site_reduction'] = {"aggregation" : "avg"}
    arbplf_ll(json.dumps(x))

def test_ok_reduction_selection():
    x = copy.deepcopy(good_input)
    x['site_reduction'] = {"selection" : [0]}
    arbplf_ll(json.dumps(x))

def test_ok_reduction_selection_aggregation():
    x = copy.deepcopy(good_input)
    x['site_reduction'] = {"selection" : [0], "aggregation" : "sum"}
    arbplf_ll(json.dumps(x))

def test_simplified_felsenstein_fig_16_4_example():
    x = {
     "model_and_data" : {
         "edges" : [[5, 0], [5, 1], [5, 6], [6, 2], [6, 7], [7, 3], [7, 4]],
         "edge_rate_coefficients" : [0.01, 0.2, 0.15, 0.3, 0.05, 0.3, 0.02],
         "rate_matrix" : [
             [0, 3, 3, 3],
             [3, 0, 3, 3],
             [3, 3, 0, 3],
             [3, 3, 3, 0]],
         "probability_array" : [[
             [1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 1, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0.25, 0.25, 0.25, 0.25],
             [1, 1, 1, 1],
             [1, 1, 1, 1]]]},
     "site_reduction" : {
         "aggregation" : "sum"}}
    arbplf_ll(json.dumps(x))
