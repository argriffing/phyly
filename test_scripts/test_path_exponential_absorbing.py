"""
Check properties at time points and on intervals between time points,
for a time series that starts in state 0
and that evolves towards absorbing state 1.

"""
from __future__ import print_function, division

from StringIO import StringIO
import json
import copy

import numpy as np
import pandas as pd
from numpy.testing import assert_equal, assert_allclose

from numpy import exp, expm1
from scipy.special import exprel

from arbplf import arbplf_dwell, arbplf_marginal, arbplf_trans

rates = [1, 1, 2, 0, 3]

D = {
        "model_and_data" : {
            "edges" : [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 5]],
            "edge_rate_coefficients" : rates,
            "rate_matrix" : [
                [0, 1],
                [0, 0]],
            "probability_array" : [[
                [1, 0],
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 1]]]
            },
        "site_reduction" : {"aggregation" : "sum"}
        }

def test_dwell():
    d = copy.deepcopy(D)
    s = arbplf_dwell(json.dumps(d))
    df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    actual = df.pivot('edge', 'state', 'value').values
    # compute the desired closed form solution
    u = np.cumsum([0] + rates)
    a, b = u[:-1], u[1:]
    v = exprel(b - a) * exp(-b)
    desired = np.vstack([v, 1-v]).T
    # compare actual and desired result
    assert_allclose(actual, desired)

def test_marginal():
    d = copy.deepcopy(D)
    s = arbplf_marginal(json.dumps(d))
    df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    actual = df.pivot('node', 'state', 'value').values
    # compute the desired closed form solution
    u = np.cumsum([0] + rates)
    desired = np.vstack([np.exp(-u), -np.expm1(-u)]).T
    # compare actual and desired result
    assert_allclose(actual, desired)

def test_trans_01():
    d = copy.deepcopy(D)
    d['trans_reduction'] = {"selection" : [[0, 1]], "aggregation" : "sum"}
    s = arbplf_trans(json.dumps(d))
    df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    actual = df.set_index('edge').value.values
    # compute the desired closed form solution
    u = np.cumsum([0] + rates)
    a, b = u[:-1], u[1:]
    desired = exp(-a) - exp(-b)
    # compare actual and desired result
    assert_allclose(actual, desired)

def test_trans_10():
    d = copy.deepcopy(D)
    d['trans_reduction'] = {"selection" : [[1, 0]], "aggregation" : "sum"}
    s = arbplf_trans(json.dumps(d))
    df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    actual = df.set_index('edge').value.values
    # compute the desired closed form solution
    desired = np.zeros_like(actual)
    # compare actual and desired result
    assert_equal(actual, desired)

def test_truncated_dwell():
    d = copy.deepcopy(D)
    d['model_and_data']['probability_array'][0][-1] = [0, 1]
    s = arbplf_dwell(json.dumps(d))
    df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    actual = df.pivot('edge', 'state', 'value').values
    # compute the desired closed form solution
    u = np.cumsum([0] + rates)
    a, b = u[:-1], u[1:]
    T = u[-1]
    # this way is not robust when a == b.
    def F(x):
        return -(exp(T - x) + x) / expm1(T)
    v = (F(b) - F(a)) / (b - a)
    desired = np.vstack([v, 1-v]).T
    # this way is better
    v = (exprel(b-a)*exp(T-b) - 1) / expm1(T)
    desired = np.vstack([v, 1-v]).T
    # compare actual and desired result
    assert_allclose(actual, desired)

def test_truncated_marginal():
    d = copy.deepcopy(D)
    d['model_and_data']['probability_array'][0][-1] = [0, 1]
    s = arbplf_marginal(json.dumps(d))
    df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    actual = df.pivot('node', 'state', 'value').values
    # compute the desired closed form solution
    u = np.cumsum([0] + rates)
    T = u[-1]
    v = (exp(-T) - exp(-u)) / expm1(-T)
    desired = np.vstack([v, 1-v]).T
    # compare actual and desired result
    assert_allclose(actual, desired)

def test_truncated_trans_01():
    d = copy.deepcopy(D)
    d['model_and_data']['probability_array'][0][-1] = [0, 1]
    d['trans_reduction'] = {"selection" : [[0, 1]], "aggregation" : "sum"}
    s = arbplf_trans(json.dumps(d))
    df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    actual = df.set_index('edge').value.values
    # compute the desired closed form solution
    u = np.cumsum([0] + rates)
    T = u[-1]
    def f(x):
        return (exp(-T) - exp(-x)) / expm1(-T)
    a, b = u[:-1], u[1:]
    desired = f(a) - f(b)
    # compare actual and desired result
    assert_allclose(actual, desired)

def test_truncated_trans_10():
    d = copy.deepcopy(D)
    d['model_and_data']['probability_array'][0][-1] = [0, 1]
    d['trans_reduction'] = {"selection" : [[1, 0]], "aggregation" : "sum"}
    s = arbplf_trans(json.dumps(d))
    df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    actual = df.set_index('edge').value.values
    # compute the desired closed form solution
    desired = np.zeros_like(actual)
    # compare actual and desired result
    assert_equal(actual, desired)
