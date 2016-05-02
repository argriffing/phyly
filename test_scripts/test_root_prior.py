"""
Test root priors on a tiny model.

The toy models and data are taken from the rate divisor test script.

"""
from __future__ import print_function, division

from StringIO import StringIO
import json
import copy
import random

import numpy as np
from numpy.testing import (
        assert_, assert_equal, assert_raises, assert_allclose, TestCase)

from arbplf import (
        arbplf_ll, arbplf_marginal,
        arbplf_dwell, arbplf_trans, arbplf_em_update,
        arbplf_newton_delta, arbplf_newton_update,
        arbplf_newton_refine, arbplf_deriv, arbplf_hess,
        arbplf_inv_hess,
        )

_funcs = (
        arbplf_ll, arbplf_marginal,
        arbplf_dwell, arbplf_trans, arbplf_em_update,
        arbplf_newton_delta, arbplf_newton_update,
        arbplf_newton_refine, arbplf_deriv, arbplf_hess,
        arbplf_inv_hess,
        )

_coefficients = [2, 13, 19]

_site_weights = [10, 1, 2, 3, 20, 2, 5, 7]

_probability_array = [
        [[0.25, 0.75],
         [1, 0],
         [1, 0],
         [1, 0]],
        [[0.25, 0.75],
         [0, 1],
         [1, 0],
         [1, 0]],
        [[0.25, 0.75],
         [1, 0],
         [0, 1],
         [1, 0]],
        [[0.25, 0.75],
         [1, 0],
         [1, 0],
         [0, 1]],
        [[0.25, 0.75],
         [0, 1],
         [0, 1],
         [0, 1]],
        [[0.25, 0.75],
         [1, 0],
         [0, 1],
         [0, 1]],
        [[0.25, 0.75],
         [0, 1],
         [1, 0],
         [0, 1]],
        [[0.25, 0.75],
         [0, 1],
         [0, 1],
         [1, 0]],
        ]

_A = {
        "model_and_data" : {
            "edges" : [[0, 1], [0, 2], [0, 3]],
            "edge_rate_coefficients" : _coefficients,
            "rate_matrix" : [
                [0, 3],
                [1, 0]],
            "rate_divisor" : 100,
            "probability_array" : _probability_array},
        "site_reduction" : {"aggregation" : _site_weights}
        }

_B = {
        "model_and_data" : {
            "edges" : [[0, 1], [0, 2], [0, 3]],
            "edge_rate_coefficients" : _coefficients,
            "rate_matrix" : [
                [0, 9],
                [3, 0]],
            "rate_divisor" : 300,
            "probability_array" : _probability_array},
        "site_reduction" : {"aggregation" : _site_weights}
        }


def test_implicit_equilibrium_root_prior():
    for f in _funcs:
        print('equilibrium test', f.__name__)

        a_in = copy.deepcopy(_A)
        for arr in a_in['model_and_data']['probability_array']:
            arr[0] = [1, 1]
        a_in['model_and_data']['root_prior'] = 'equilibrium_distribution'

        b_in = copy.deepcopy(_B)
        for arr in b_in['model_and_data']['probability_array']:
            arr[0] = [0.25, 0.75]

        if f is arbplf_trans:
            trans = {
                "selection" : [[0, 1], [1, 0]],
                "aggregation" : "sum"}
            a_in['trans_reduction'] = trans
            b_in['trans_reduction'] = trans

        a_out = json.loads(f(json.dumps(a_in)))
        b_out = json.loads(f(json.dumps(b_in)))

        assert_equal(a_out, b_out)

def test_explicit_root_prior():
    for f in _funcs:
        print('equilibrium test', f.__name__)

        a_in = copy.deepcopy(_A)
        for arr in a_in['model_and_data']['probability_array']:
            arr[0] = [1, 1]
        a_in['model_and_data']['root_prior'] = 'equilibrium_distribution'

        b_in = copy.deepcopy(_B)
        for arr in b_in['model_and_data']['probability_array']:
            arr[0] = [1, 1]
        b_in['model_and_data']['root_prior'] = [0.25, 0.75]

        if f is arbplf_trans:
            trans = {
                "selection" : [[0, 1], [1, 0]],
                "aggregation" : "sum"}
            a_in['trans_reduction'] = trans
            b_in['trans_reduction'] = trans

        a_out = json.loads(f(json.dumps(a_in)))
        b_out = json.loads(f(json.dumps(b_in)))

        assert_equal(a_out, b_out)
