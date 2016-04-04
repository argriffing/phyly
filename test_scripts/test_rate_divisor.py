"""
Test rate divisors on a tiny model.

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
        arbplf_dwell, arbplf_trans, arbplf_coeff_expect,
        arbplf_newton_delta, arbplf_newton_point,
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


def test_rate_divisor():

    for f in (
            arbplf_ll, arbplf_marginal,
            arbplf_dwell, arbplf_trans, arbplf_coeff_expect,
            arbplf_newton_delta, arbplf_newton_point,
            arbplf_newton_refine, arbplf_deriv, arbplf_hess,
            arbplf_inv_hess,
            ):

        a_in = copy.deepcopy(_A)
        b_in = copy.deepcopy(_B)

        if f is arbplf_trans:
            trans = {
                "selection" : [[0, 1], [1, 0]],
                "aggregation" : "sum"}
            a_in['trans_reduction'] = trans
            b_in['trans_reduction'] = trans

        a_out = json.loads(f(json.dumps(a_in)))
        b_out = json.loads(f(json.dumps(b_in)))

        assert_equal(a_out, b_out)


def test_equilibrium():
    for f in (
            arbplf_ll, arbplf_marginal,
            arbplf_dwell, arbplf_trans, arbplf_coeff_expect,
            arbplf_newton_delta, arbplf_newton_point,
            arbplf_newton_refine, arbplf_deriv, arbplf_hess,
            arbplf_inv_hess,
            ):

        print('equilibrium test', f.__name__)

        a_in = copy.deepcopy(_A)
        for arr in a_in['model_and_data']['probability_array']:
            arr[0] = [1, 1]
        a_in['model_and_data']['use_equilibrium_root_prior'] = True

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


def test_rate_matrix_diagonal_entries():
    # diagonal entries should be ignored

    for f in (
            arbplf_ll, arbplf_marginal,
            arbplf_dwell, arbplf_trans, arbplf_coeff_expect,
            arbplf_newton_delta, arbplf_newton_point,
            arbplf_newton_refine, arbplf_deriv, arbplf_hess,
            arbplf_inv_hess,
            ):

        print('equilibrium test', f.__name__)

        _C = copy.deepcopy(_A)
        for arr in _C['model_and_data']['probability_array']:
            arr[0] = [1, 1]
        _C['model_and_data']['use_equilibrium_root_prior'] = True

        a_in = copy.deepcopy(_C)

        b_in = copy.deepcopy(_C)
        for i in range(2):
            b_in['model_and_data']['rate_matrix'][i][i] = 42

        if f is arbplf_trans:
            trans = {
                "selection" : [[0, 1], [1, 0]],
                "aggregation" : "sum"}
            a_in['trans_reduction'] = trans
            b_in['trans_reduction'] = trans

        a_out = json.loads(f(json.dumps(a_in)))
        b_out = json.loads(f(json.dumps(b_in)))

        assert_equal(a_out, b_out)


_C = {
        "model_and_data" : {
            "edges" : [[0, 1], [0, 2], [0, 3]],
            "edge_rate_coefficients" : [c / 100 for c in _coefficients],
            "rate_matrix" : [
                [0, 3],
                [1, 0]],
            "rate_divisor" : 1.5,
            "probability_array" : _probability_array},
        "site_reduction" : {"aggregation" : _site_weights}
        }

_D = {
        "model_and_data" : {
            "edges" : [[0, 1], [0, 2], [0, 3]],
            "edge_rate_coefficients" : [c / 100 for c in _coefficients],
            "rate_matrix" : [
                [0, 9],
                [3, 0]],
            'rate_divisor' : 'equilibrium_exit_rate_expectation',
            "probability_array" : _probability_array},
        "site_reduction" : {"aggregation" : _site_weights}
        }


def test_equilibrium_rate_divisor():
    for f in (
            arbplf_ll, arbplf_marginal,
            arbplf_newton_delta, arbplf_newton_point,
            arbplf_newton_refine, arbplf_deriv, arbplf_hess, arbplf_inv_hess,
            arbplf_dwell, arbplf_trans, arbplf_coeff_expect,
            ):

        print('equilibrium rate divisor test', f.__name__)

        c_in = copy.deepcopy(_C)
        for arr in c_in['model_and_data']['probability_array']:
            arr[0] = [0.25, 0.75]

        d_in = copy.deepcopy(_D)
        for arr in c_in['model_and_data']['probability_array']:
            arr[0] = [0.25, 0.75]

        if f is arbplf_trans:
            trans = {
                "selection" : [[0, 1], [1, 0]],
                "aggregation" : "sum"}
            c_in['trans_reduction'] = trans
            d_in['trans_reduction'] = trans

        print(c_in)
        print(d_in)
        print()

        c_out = json.loads(f(json.dumps(c_in)))
        d_out = json.loads(f(json.dumps(d_in)))

        assert_equal(c_out, d_out)
