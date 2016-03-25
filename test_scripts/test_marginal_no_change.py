"""
Test a very simple model.

The tree is a path like 0--1--2,
and the process is a pure-birth process.
In this test we look at what happens in node 1 when
the state of 0 and the state of 2 are both known to be 'unborn'.
The true conditional marginal state distribution at node 1 is (1, 0); that is,
we know with complete certainty that node 1 is in the 'unborn' state.
Also, we know that the log likelihood of this system should be exactly -2,
corresponding to no event having occurred in either of two segments
each with poisson rate 1 of the event occurring.

"""
from __future__ import print_function, division

from StringIO import StringIO
import json
import copy

import numpy as np
from numpy.testing import (
        assert_, assert_equal, assert_raises, assert_allclose, TestCase)

from arbplf import arbplf_ll
from arbplf import arbplf_marginal

def test_marginal_no_change():
    d = {
            "model_and_data" : {
                "edges" : [[0, 1], [1, 2]],
                "edge_rate_coefficients" : [1, 1],
                "rate_matrix" : [[0, 1], [0, 0]],
                "probability_array" : [[[1, 0], [1, 1], [1, 0]]]
                }
            }

    actual_marginal = json.loads(arbplf_marginal(json.dumps(d)))
    desired_marginal = {
        "columns": ["site", "node", "state", "value"],
        "data": [
            [0, 0, 0, 1.0],
            [0, 0, 1, 0.0],
            [0, 1, 0, 1.0],
            [0, 1, 1, 0.0],
            [0, 2, 0, 1.0],
            [0, 2, 1, 0.0]]
        }
    assert_equal(actual_marginal, desired_marginal)

    actual_ll = json.loads(arbplf_ll(json.dumps(d)))
    desired_ll = {
        "columns": ["site", "value"],
        "data": [[0, -2.0]]
        }
    assert_equal(actual_ll, desired_ll)


def test_marginal_no_change_heterogeneous_edge_rates():
    # try changing one of the edge rate coefficients
    d = {
            "model_and_data" : {
                "edges" : [[0, 1], [1, 2]],
                "edge_rate_coefficients" : [1, 2],
                "rate_matrix" : [[0, 1], [0, 0]],
                "probability_array" : [[[1, 0], [1, 1], [1, 0]]]
                }
            }

    actual_marginal = json.loads(arbplf_marginal(json.dumps(d)))
    desired_marginal = {
        "columns": ["site", "node", "state", "value"],
        "data": [
            [0, 0, 0, 1.0],
            [0, 0, 1, 0.0],
            [0, 1, 0, 1.0],
            [0, 1, 1, 0.0],
            [0, 2, 0, 1.0],
            [0, 2, 1, 0.0]]
        }
    assert_equal(actual_marginal, desired_marginal)

    actual_ll = json.loads(arbplf_ll(json.dumps(d)))
    desired_ll = {
        "columns": ["site", "value"],
        "data": [[0, -3.0]]
        }
    assert_equal(actual_ll, desired_ll)


def test_marginal_no_change_edges_are_not_preordered():
    # Try switching the order of the edges in the input
    # and increasing the birth rate in the rate matrix.
    d = {
            "model_and_data" : {
                "edges" : [[1, 2], [0, 1]],
                "edge_rate_coefficients" : [1, 2],
                "rate_matrix" : [[0, 2], [0, 0]],
                "probability_array" : [[[1, 0], [1, 1], [1, 0]]]
                }
            }

    actual_marginal = json.loads(arbplf_marginal(json.dumps(d)))
    desired_marginal = {
        "columns": ["site", "node", "state", "value"],
        "data": [
            [0, 0, 0, 1.0],
            [0, 0, 1, 0.0],
            [0, 1, 0, 1.0],
            [0, 1, 1, 0.0],
            [0, 2, 0, 1.0],
            [0, 2, 1, 0.0]]
        }
    assert_equal(actual_marginal, desired_marginal)

    actual_ll = json.loads(arbplf_ll(json.dumps(d)))
    desired_ll = {
        "columns": ["site", "value"],
        "data": [[0, -6.0]]
        }
    assert_equal(actual_ll, desired_ll)
