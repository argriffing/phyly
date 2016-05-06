"""
Test marginal distribution reductions.

"""
from __future__ import print_function, division

from StringIO import StringIO
import json
import copy

import numpy as np
import pandas as pd
from numpy.testing import (
        assert_, assert_equal, assert_raises, assert_allclose, TestCase)

from arbplf import arbplf_ll
from arbplf import arbplf_marginal


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

def _assert_allclose_series(a, b):
    # Align the layouts of the two pandas Series objects,
    # preparing to compare their values.
    # The align member function returns two new pandas Series objects.
    u, v = a.align(b)
    assert_allclose(u.values, v.values)

def myll(d):
    """
    Provides a dict -> pandas.DataFrame wrapper of the JSON arbplf_ll.
    """
    s = arbplf_ll(json.dumps(d))
    df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    return df

def mymarginal(d):
    """
    Provides a dict -> pandas.DataFrame wrapper of the JSON arbplf_marginal.
    """
    s = arbplf_marginal(json.dumps(d))
    df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    return df

def test_default_marginal():
    mymarginal(default_in)

def test_default_marginal_vs_empty_reductions():
    f = mymarginal
    x = copy.deepcopy(default_in)
    x['site_reduction'] = {}
    x['node_reduction'] = {}
    x['state_reduction'] = {}
    v = f(default_in)
    u = f(x)
    assert_allclose(u.values, v.values)


def test_sum_over_sites():
    # see comments in test_deriv_sum_over_sites for more detail

    full = mymarginal(default_in)

    # Average over all axes *except* 'node' and 'state'.
    pd_sum = full.groupby(['node', 'state']).value.mean()

    # Use the more accurate and efficient built-in aggregation mechanism.
    x = copy.deepcopy(default_in)
    x['site_reduction'] = {'aggregation' : 'avg'}
    x['node_reduction'] = {'selection' : [4, 5, 6, 7, 3, 2, 1, 0]}
    phyly_sum = mymarginal(x).set_index(['node', 'state']).value

    # These pandas Series objects should contain the same information,
    # up to small numerical errors, but the underlying arrays are not
    # directly comparable because the internal layouts are different.
    assert_(not np.allclose(pd_sum.values, phyly_sum.values))

    _assert_allclose_series(pd_sum, phyly_sum)


def test_weighted_sum_over_states():
    full = mymarginal(default_in)
    state_weights = [0.1, 0.2, 0.3, 0.4]

    # weighted sum over states
    s = full.copy()
    s['value'] = s['value'] * [state_weights[x] for x in s['state']]
    pd_sum = s.groupby(['node']).sum().value

    # Use the more accurate and efficient built-in aggregation mechanism.
    x = copy.deepcopy(default_in)
    x['site_reduction'] = {'aggregation' : 'sum'}
    x['state_reduction'] = {'aggregation' : state_weights}
    phyly_sum = mymarginal(x).set_index('node').value

    _assert_allclose_series(pd_sum, phyly_sum)


def test_via_likelihood():
    # A marginal state distribution at a node can be computed
    # inefficiently using only a likelihood calculation.
    # Check that this inefficient likelihood-only method gives the same
    # result as the more efficient method.

    # Pick an arbitrary site and an arbitrary node.
    node = 6
    site = 1

    # Compute marginal probabilities for 4 states at the node and site.
    x = copy.deepcopy(default_in)
    x['node_reduction'] = {'selection' : [node], 'aggregation' : 'sum'}
    x['site_reduction'] = {'selection' : [site], 'aggregation' : 'sum'}
    distn_via_marginal = mymarginal(x).set_index('state').value

    # Compute log likelihoods for 4 states at the node and site.
    # Use the log likelihoods to define a distribution over the states.
    y = copy.deepcopy(default_in)
    arr = []
    for i in range(4):
        m = copy.deepcopy(
                default_in['model_and_data']['probability_array'][site])
        m[node] = [0, 0, 0, 0]
        m[node][i] = 1
        arr.append(m)
    y['model_and_data']['probability_array'] = arr
    series_ll = myll(y).set_index('site').value
    lhoods = series_ll.apply(np.exp)
    distn_via_likelihood = lhoods / lhoods.sum()

    _assert_allclose_series(distn_via_marginal, distn_via_likelihood)

def test_rates_across_sites_invariant_a():
    v = mymarginal(default_in)
    # because all the rates are 1, the output should not be affected
    rates = [1, 1, 1, 1]
    for prior in [0.1, 0.2, 0.3, 0.4], 'uniform_distribution':
        x = copy.deepcopy(default_in)
        x['model_and_data']['rate_mixture'] = dict(rates=rates, prior=prior)
        u = mymarginal(x)
        assert_allclose(u.values, v.values)

def test_rates_across_sites_invariant_b():
    v = mymarginal(default_in)
    # none of the original sites are invariant,
    # so the output should not be affected
    rates = [0, 1]
    for prior in [0.4, 0.6], 'uniform_distribution':
        x = copy.deepcopy(default_in)
        x['model_and_data']['rate_mixture'] = dict(rates=rates, prior=prior)
        u = mymarginal(x)
        assert_allclose(u.values, v.values)

def test_rates_across_sites_invariant_c():
    v = mymarginal(default_in)
    # the only rate category with a nonzero rate*probability
    # has a rate of 1, so the output should not be affected
    rates = [0, 1, 2]
    prior = [0.2, 0.8, 0.0]
    x = copy.deepcopy(default_in)
    x['model_and_data']['rate_mixture'] = dict(rates=rates, prior=prior)
    u = mymarginal(x)
    assert_allclose(u.values, v.values)

def test_rates_across_sites_invariant_d():
    # Same as invariant_c except with reductions.
    # Pick an arbitrary site and an arbitrary node.
    node = 6
    site = 1
    #
    x = copy.deepcopy(default_in)
    x['node_reduction'] = {'selection' : [node], 'aggregation' : 'sum'}
    x['site_reduction'] = {'selection' : [site], 'aggregation' : 'sum'}
    v = mymarginal(x)
    # the only rate category with a nonzero rate*probability
    # has a rate of 1, so the output should not be affected
    rates = [0, 1, 2]
    prior = [0.2, 0.8, 0.0]
    x = copy.deepcopy(default_in)
    x['node_reduction'] = {'selection' : [node], 'aggregation' : 'sum'}
    x['site_reduction'] = {'selection' : [site], 'aggregation' : 'sum'}
    x['model_and_data']['rate_mixture'] = dict(rates=rates, prior=prior)
    u = mymarginal(x)
    assert_allclose(u.values, v.values)

def test_rates_across_sites_uniform():
    rates = [1, 2, 3]
    #
    prior = 'uniform_distribution'
    x = copy.deepcopy(default_in)
    x['model_and_data']['rate_mixture'] = dict(rates=rates, prior=prior)
    u = mymarginal(x)
    #
    prior = [1/3, 1/3, 1/3]
    x = copy.deepcopy(default_in)
    x['model_and_data']['rate_mixture'] = dict(rates=rates, prior=prior)
    v = mymarginal(x)
    #
    assert_allclose(u.values, v.values)

def test_rates_across_sites_vs_block_diagonal():
    # Compare rates across sites vs. a block diagonal workaround.
    rates = [1, 2]
    prior = [0.25, 0.75]
    # Pick an arbitrary site and an arbitrary node.
    node = 6
    site = 1
    #
    x = copy.deepcopy(default_in)
    x['model_and_data']['rate_matrix'] = [
            [0., .3, .4, .5, 0., 0., 0., 0.],
            [.3, 0., .3, .3, 0., 0., 0., 0.],
            [.3, .6, 0., .3, 0., 0., 0., 0.],
            [.3, .3, .3, 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., .6, .8, 1.],
            [0., 0., 0., 0., .6, 0., .6, .6],
            [0., 0., 0., 0., .6, 1.2, 0., .6],
            [0., 0., 0., 0., .6, .6, .6, .0]]
    x['model_and_data']['probability_array'] = [
            [
            [1, 0, 0, 0,   1, 0, 0, 0],
            [0, 1, 0, 0,   0, 1, 0, 0],
            [0, 1, 0, 0,   0, 1, 0, 0],
            [0, 1, 0, 0,   0, 1, 0, 0],
            [0, 0, 1, 0,   0, 0, 1, 0],
            [0.0625, 0.0625, 0.0625, 0.0625, 0.1875, 0.1875, 0.1875, 0.1875],
            [1, 1, 1, 1,   1, 1, 1, 1],
            [1, 1, 1, 1,   1, 1, 1, 1]],
            [
            [1, 0, 0, 0,   1, 0, 0, 0],
            [0, 1, 0, 0,   0, 1, 0, 0],
            [0, 0, 0, 1,   0, 0, 0, 1],
            [0, 1, 0, 0,   0, 1, 0, 0],
            [0, 0, 1, 0,   0, 0, 1, 0],
            [0.0625, 0.0625, 0.0625, 0.0625, 0.1875, 0.1875, 0.1875, 0.1875],
            [1, 1, 1, 1,   1, 1, 1, 1],
            [1, 1, 1, 1,   1, 1, 1, 1]]]
    x['node_reduction'] = {'selection' : [node], 'aggregation' : 'sum'}
    x['site_reduction'] = {'selection' : [site], 'aggregation' : 'sum'}
    u = mymarginal(x).set_index('state').value.sort_index().values
    #
    x = copy.deepcopy(default_in)
    x['model_and_data']['rate_mixture'] = dict(rates=rates, prior=prior)
    x['node_reduction'] = {'selection' : [node], 'aggregation' : 'sum'}
    x['site_reduction'] = {'selection' : [site], 'aggregation' : 'sum'}
    v = mymarginal(x).set_index('state').value.sort_index().values
    #
    assert_allclose(u[:4] + u[4:], v)
