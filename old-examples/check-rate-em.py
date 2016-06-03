"""
For a rate matrix with expected rate 1,
compare maximum likelihood edge rate coefficients vs.
the per-edge conditionally expected number of transitions
under the maximum likelihood edge rate estimates.

"""
from __future__ import print_function, division

from StringIO import StringIO
import json
import copy
import random
import pandas as pd

import numpy as np
from numpy.testing import (
        assert_, assert_equal, assert_raises, assert_allclose, TestCase)

from arbplf import (
        arbplf_ll, arbplf_trans, arbplf_em_update, arbplf_newton_refine)

def equilibrium(Q):
    n = Q.shape[0]
    R = np.zeros((n+1, n+1))
    R[:-1, :-1] = Q.T
    R[-1, :-1] = 1
    R[:-1, -1] = 1
    x = np.linalg.solve(R, np.ones(n+1))
    p = x[:-1]
    return p

def sample_rate_matrix(n):
    Q = np.exp(np.random.randn(n, n))
    Q = Q - np.diag(Q.sum(axis=1))
    return Q

def sample_reversible_rate_matrix(n):
    A = np.random.randn(n, n)
    Q = np.exp(A + A.T)
    D = np.diag(np.exp(np.random.randn(n)))
    Q = Q.dot(D)
    Q = Q - np.diag(Q.sum(axis=1))
    return Q

def main():
    np.random.seed(123475)

    # sample a random rate matrix
    state_count = 3
    edge_count = 3
    node_count = edge_count + 1
    #Q = sample_rate_matrix(state_count)
    Q = sample_reversible_rate_matrix(state_count)
    p = equilibrium(Q)
    expected_rate = -p.dot(np.diag(Q))
    print('expected rate:', expected_rate)
    Q = Q / expected_rate
    np.fill_diagonal(Q, 0)
    # use ad hoc data
    probability_array = [
            [[1, 1, 1],
             [1, 0, 0],
             [1, 0, 0],
             [1, 0, 0]],
            [[1, 1, 1],
             [0, 1, 0],
             [1, 0, 0],
             [1, 0, 0]],
            [[1, 1, 1],
             [1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]]
    site_weights = [.7, .2, .1]
    edges = [[0, 1], [0, 2], [0, 3]]
    coefficients = [.01, .01, .01]
    d = {
        "model_and_data" : {
            "edges" : edges,
            "edge_rate_coefficients" : coefficients,
            "rate_matrix" : Q.tolist(),
            "probability_array" : probability_array},
        "site_reduction" : {"aggregation" : site_weights}
        }
    print(d)
    for i in range(100):
        s = arbplf_em_update(json.dumps(d))
        df = pd.read_json(StringIO(s), orient='split', precise_float=True)
        y = df.value.values.tolist()
        d['model_and_data']['edge_rate_coefficients'] = y
        print('coefficients updated by EM:', y)
    s = arbplf_newton_refine(json.dumps(d))
    df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    y = df.value.values.tolist()
    print('coefficients updated by newton refinement:', y)

    d['trans_reduction'] = {
            'selection' : [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]],
            'aggregation' : 'sum'}
    d['model_and_data']['edge_rate_coefficients'] = y

    s = arbplf_trans(json.dumps(d))
    df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    y = df.value.values.tolist()
    print('conditionally expected transition counts:', y)

main()
