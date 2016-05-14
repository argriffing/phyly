"""
Counting labeled transitions in continuous-time Markov models of evolution
by Minin and Suchard

"""
from __future__ import print_function, division

from StringIO import StringIO
import json
import itertools
import math

import numpy as np
import pandas as pd

from arbplf import arbplf_ll, arbplf_trans

#def mytrans(d):
    #s = arbplf_dwell(json.dumps(d))
    #df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    #return df

def gen_K80():
    # Use the nucleotide order from the Minin and Suchard paper.
    # A, G, C, T
    transitions = ((0, 1), (1, 0), (2, 3), (3, 2))
    for i in range(4):
        for j in range(4):
            if i != j:
                if (i, j) in transitions:
                    ts, tv = 1, 0
                else:
                    ts, tv = 0, 1
                yield i, j, ts, tv

def get_rate_matrix(kappa):
    # Stationary distribution is uniform.
    # Each row has one transition and two transversions,
    # so an expected rate before normalization is kappa + 2.
    m = [[0]*4 for i in range(4)]
    expected_rate = kappa + 2
    for i, j, ts, tv in gen_K80():
        m[i][j] = kappa * ts + tv
    return m, expected_rate

def get_ts_tv_pairs():
    ts_pairs = []
    tv_pairs = []
    for i, j, ts, tv in gen_K80():
        pair = [i, j]
        if ts:
            ts_pairs.append(pair)
        if tv:
            tv_pairs.append(pair)
    return ts_pairs, tv_pairs

def run(assumed_kappa):
    state_count = 4
    node_count = 5
    true_kappa = 4
    assumed_m, assumed_denom = get_rate_matrix(assumed_kappa)
    true_m, true_denom = get_rate_matrix(true_kappa)
    edges = [[0, 2], [0, 1], [1, 3], [1, 4]]
    assumed_coeffs = [28, 21, 12, 9]
    true_coeffs = [30, 20, 10, 10]
    # There are five nodes.
    # Three of them have unobserved states.
    # Use one site for each of the 4^3 = 64 possible observations.
    X = [-1]
    U = range(4)
    all_site_patterns = list(itertools.product(X, X, U, U, U))
    prior_array = [[
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
            "edge_rate_coefficients" : true_coeffs,
            "root_prior" : "equilibrium_distribution",
            "rate_matrix" : true_m,
            "rate_divisor" : true_denom * 100,
            "probability_array" : probability_array}
    d = {"model_and_data" : model_and_data}
    s = arbplf_ll(json.dumps(d))
    df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    log_likelihoods = df.value.values
    print('log likelihood sum:', sum(log_likelihoods))

    # compute ts and tv using the likelihoods as observation weights
    weights = [math.exp(ll) for ll in log_likelihoods]
    total = sum(weights)
    weights = [w / total for w in weights]
    ts_pairs, tv_pairs = get_ts_tv_pairs()

    model_and_data = {
            "edges" : edges,
            "edge_rate_coefficients" : assumed_coeffs,
            "root_prior" : "equilibrium_distribution",
            "rate_matrix" : assumed_m,
            "rate_divisor" : assumed_denom * 100,
            "probability_array" : probability_array}
    d = {
            "model_and_data" : model_and_data,
            "site_reduction" : {"aggregation" : weights},
            "edge_reduction" : {"aggregation" : "sum"},
            "trans_reduction" : {"aggregation" : "sum"}}

    d['trans_reduction']['selection'] = ts_pairs
    d['trans_reduction']['aggregation'] = [1000]*len(ts_pairs)
    d['site_reduction']['aggregation'] = "sum"
    d['model_and_data']['probability_array'] = prior_array
    s = arbplf_trans(json.dumps(d))
    df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    print("prior ts expectation:")
    print(df.value.values[0])
    print(s)

    d['trans_reduction']['selection'] = ts_pairs
    d['trans_reduction']['aggregation'] = [1000]*len(ts_pairs)
    d['site_reduction']['aggregation'] = weights
    d['model_and_data']['probability_array'] = probability_array
    s = arbplf_trans(json.dumps(d))
    df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    print("conditional ts expectation:")
    print(df.value.values[0])
    print(s)

    d['trans_reduction']['selection'] = tv_pairs
    d['trans_reduction']['aggregation'] = [1000]*len(tv_pairs)
    d['site_reduction']['aggregation'] = "sum"
    d['model_and_data']['probability_array'] = prior_array
    s = arbplf_trans(json.dumps(d))
    df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    print("prior tv expectation:")
    print(df.value.values[0])
    print(s)

    d['trans_reduction']['selection'] = tv_pairs
    d['trans_reduction']['aggregation'] = [1000]*len(tv_pairs)
    d['site_reduction']['aggregation'] = weights
    d['model_and_data']['probability_array'] = probability_array
    s = arbplf_trans(json.dumps(d))
    df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    print("conditional tv expectation:")
    print(df.value.values[0])
    print(s)

def main():
    for kappa in 1, 2, 4:
        print("kappa:", kappa)
        run(kappa)
        print()

main()
