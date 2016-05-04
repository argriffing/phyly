"""
"""
from __future__ import print_function, division

from StringIO import StringIO
import json
import itertools
import math

import numpy as np
import pandas as pd
from scipy.linalg import expm

from arbplf import arbplf_ll, arbplf_trans

def make_rate_matrix(a, b, c, d, e, pA, pC, pG, pT):
    # try to use the Bio++ GTR parameterization
    # p = 2 * (a*pC*pT + b*pA*pT + c*pG*pT + d*pA*pC + e*pC*pG+ pA*pG)
    S = [
            [0, d, 1, b],
            [d, 0, e, a],
            [1, e, 0, c],
            [b, a, c, 0]]
    for j, p in enumerate((pA, pC, pG, pT)):
        for i in range(4):
            S[i][j] *= p
    Q = np.array(S)

    # check the rate matrix
    p = np.array([pA, pC, pG, pT])
    #Q = Q * p
    #Q = Q / p[np.newaxis, :]
    #Q = Q * np.sqrt(p) / np.sqrt(p)[np.newaxis, :]
    r = Q.sum(axis=1)
    Q = Q - np.diag(r)
    Q = Q / np.dot(-np.diag(Q), p)
    t = np.array([
    [-1.37306, 0.362694, 0.906736, 0.103627],
    [0.103627, -0.984456, 0.362694, 0.518135],
    [0.259067, 0.362694, -0.777202, 0.15544],
    [0.0518135, 0.906736, 0.272021, -1.23057],
    ])
    print('test rate matrix:')
    print(np.dot(-np.diag(t), p))
    print(t.sum(axis=1))
    print(t)
    print('reconstructed rate matrix:')
    print(np.dot(-np.diag(Q), p))
    print(Q.sum(axis=1))
    print(Q)

    return S
    

def run():
    state_count = 4
    edge_count = 5
    node_count = edge_count + 1

    # Define the tree used in the phyl transition mapping example.
    edges = [[4, 0], [4, 1], [5, 4], [5, 2], [5, 3]]
    inference_rates = [0.001, 0.002, 0.008, 0.01, 0.1]
    simulation_rates = [0.001 * (9 / 20), 0.002, 0.008, 0.01, 0.1]

    """
    # Define the poisson rate matrix with expected exit rate 1
    rate_divisor = 3
    rate_matrix = [
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0]]
    """
    # use a GTR rate matrix
    a, b, c, d, e, pA, pC, pG, pT = (
            1, 0.2, 0.3, 0.4, 0.4, 0.1, 0.35, 0.35, 0.2)
    rate_matrix = make_rate_matrix(a, b, c, d, e, pA, pC, pG, pT)

    # Use one site for each of the 4^4 = 256 possible observations.
    X = [-1]
    U = range(4)
    all_site_patterns = list(itertools.product(U, U, U, U, X, X))
    prior_array = [[
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]]]
    probability_array = []
    for pattern in all_site_patterns:
        arr = []
        for i, p in enumerate(pattern):
            if p == -1:
                row = [1]*state_count
            else:
                row = [0]*state_count
                row[p] = 1
            arr.append(row)
        probability_array.append(arr)
    model_and_data = {
            "edges" : edges,
            "edge_rate_coefficients" : simulation_rates,
            "rate_divisor" : "equilibrium_exit_rate",
            "root_prior" : "equilibrium_distribution",
            "rate_matrix" : rate_matrix,
            "probability_array" : probability_array}
    d = {"model_and_data" : model_and_data}
    s = json.dumps(d)
    s = arbplf_ll(s)
    df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    log_likelihoods = df.value.values

    # compute expectations using the likelihoods as observation weights
    weights = [math.exp(ll) for ll in log_likelihoods]
    total = sum(weights)
    weights = [(20000 * w) / total for w in weights]

    model_and_data = {
            "edges" : edges,
            "edge_rate_coefficients" : inference_rates,
            "rate_divisor" : "equilibrium_exit_rate",
            "root_prior" : "equilibrium_distribution",
            "rate_matrix" : rate_matrix,
            "probability_array" : probability_array}
    d = {
            "model_and_data" : model_and_data,
            "site_reduction" : {"aggregation" : weights},
            "trans_reduction" : {"aggregation" : "sum"}}

    d['model_and_data']['probability_array'] = prior_array
    d['trans_reduction']['selection'] = [
            [i, j] for i in range(4) for j in range(4) if i != j]
    d['site_reduction'] = {"aggregation" : "sum"}
    s = arbplf_trans(json.dumps(d))
    df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    print("prior expectation:")
    print(20000 * df.value.values)

    d['model_and_data']['probability_array'] = probability_array
    d['trans_reduction']['selection'] = [
            [i, j] for i in range(4) for j in range(4) if i != j]
    d['site_reduction'] = {"aggregation" : weights}
    s = arbplf_trans(json.dumps(d))
    df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    print("conditional expectation:")
    print(df.value.values)

def main():
    run()

main()
