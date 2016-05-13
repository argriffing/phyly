"""
Test EM monotonicity with small random models.

"""
from __future__ import print_function, division

from StringIO import StringIO
import json
import copy

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_less

from arbplf import arbplf_ll, arbplf_em_update

def _assert_allclose_series(a, b):
    # Align the layouts of the two pandas Series objects,
    # preparing to compare their values.
    # The align member function returns two new pandas Series objects.
    u, v = a.align(b)
    assert_allclose(u.values, v.values)

def _df(f, d):
    s = f(json.dumps(d))
    df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    return df

def sample_rate_matrix(n):
    # Sample nonnegative off-diagonal entries of a rate matrix.
    scale = np.random.exponential()
    rate_matrix = scale * np.exp(np.random.randn(n, n))
    np.fill_diagonal(rate_matrix, 0)
    return rate_matrix

def sample_rate_mixture():
    # Sample a rate mixture.
    ncats = np.random.randint(1, 5)
    mixture_rates = np.exp(np.random.randn(ncats))
    mixture_prior = np.exp(np.random.randn(ncats))
    mixture_prior = mixture_prior / np.sum(mixture_prior)
    return dict(rates=mixture_rates.tolist(), prior=mixture_prior.tolist())

def check_em_monotonicity():
    # The tree is fixed with four nodes and three branches.
    edges = [[0, 1], [0, 2], [0, 3]]
    edge_count = len(edges)
    node_count = edge_count + 1

    # Sample a random number of states.
    n = np.random.randint(2, 5)

    # Sample nonnegative off-diagonal entries of a rate matrix.
    rate_matrix = sample_rate_matrix(n)

    # Sample nonnegative edge rate scaling factors.
    edge_rates_in = np.exp(np.random.randn(edge_count))

    # Sample an array of soft observations.
    site_count = np.random.randint(1, 5)
    arr = np.exp(np.random.randn(site_count, node_count, n))
    arr = arr / arr.sum(axis=1)[:, np.newaxis]

    # Define a random model and data.
    m = {}
    m['edges'] = edges
    m['edge_rate_coefficients'] = edge_rates_in.tolist()
    m['rate_matrix'] = rate_matrix.tolist()
    m['probability_array'] = arr.tolist()

    # Maybe use a mixture of rates.
    rate_mixture = sample_rate_mixture()
    if np.random.randint(2):
        m['rate_mixture'] = rate_mixture

    # ask for the log likelihood
    a_in = {'model_and_data' : m, 'site_reduction' : {"aggregation" : "sum"}}
    df = _df(arbplf_ll, a_in)
    initial_ll = df.value[0]

    # ask for the EM update of edge rate coefficients
    a_in = {'model_and_data' : m, 'site_reduction' : {"aggregation" : "sum"}}
    df = _df(arbplf_em_update, a_in)
    em_rates = df.value.values

    # ask for the log likelihood with updated edge rate scaling factors
    b_in = copy.deepcopy(a_in)
    b_in['model_and_data']['edge_rate_coefficients'] = em_rates.tolist()
    df = _df(arbplf_ll, b_in)
    final_ll = df.value[0]

    assert_array_less(initial_ll, final_ll)


def test_em_monotonicity():
    np.random.seed(1234)
    for i in range(10):
        check_em_monotonicity()

# 7 edges
# 2 sites
default_in = {
 "model_and_data" : {
     "edges" : [[5, 0], [5, 1], [5, 6], [6, 2], [6, 7], [7, 3], [7, 4]],
     "edge_rate_coefficients" : [0.01, 0.2, 0.15, 0.3, 0.05, 0.3, 0.02],
     "rate_matrix" : [
         [0, .3, .4, .5],
         [.3, 0, .3, .3],
         [.3, .6, 0, .3],
         [.3, .3, .3, 0]],
     "probability_array" : [
         [
         [1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 1, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0.25, 0.25, 0.25, 0.25],
         [1, 1, 1, 1],
         [1, 1, 1, 1]],
         [
         [1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0.25, 0.25, 0.25, 0.25],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]]},
     }

def test_degenerate_rate_mixtures():
    # Note that for this example,
    # a category with rate equal to zero has zero conditional probability.
    x = copy.deepcopy(default_in)
    x['site_reduction'] = dict(aggregation='sum')
    v = _df(arbplf_em_update, x).set_index('edge').value
    degenerate_rate_mixtures = (
            dict(rates=[1, 1, 1, 1], prior=[0.1, 0.2, 0.3, 0.4]),
            dict(rates=[1, 1, 1, 1], prior='uniform_distribution'),
            dict(rates=[1, 2], prior=[1, 0]),
            dict(rates=[3, 1], prior=[0, 1]),
            dict(rates=[0, 1], prior=[0.4, 0.6]),
            dict(rates=[0, 1], prior='uniform_distribution'),
            dict(rates=[0, 1, 2], prior=[0.2, 0.8, 0.0]),
            )
    x = copy.deepcopy(default_in)
    x['site_reduction'] = dict(aggregation='sum')
    for rate_mixture in degenerate_rate_mixtures:
        x['model_and_data']['rate_mixture'] = rate_mixture
        u = _df(arbplf_em_update, x).set_index('edge').value
        assert_allclose(u.values, v.values)
