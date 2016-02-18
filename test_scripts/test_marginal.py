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
from arbplf import arbplf_deriv
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

def myderiv(d):
    """
    Provides a dict -> pandas.DataFrame wrapper of the pure JSON arbplf_deriv.
    """
    s = arbplf_deriv(json.dumps(d))
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


def test_marginal_sum_over_sites():
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


def test_marginal_via_likelihood():
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



"""

class CheckDeriv(object):

    def __init__(self):
        self.f = myderiv(default_in)

    def setUp(self):
        self.x = copy.deepcopy(default_in)
        self.y = copy.deepcopy(default_in)


class CheckDerivAggregation(CheckDeriv, TestCase):

    def __init__(self, *args, **kwargs):
        TestCase.__init__(self, *args, **kwargs)
        CheckDeriv.__init__(self)

    def test_sum_over_sites(self):
        f = self.f.groupby('edge').value.sum()
        self.x['site_reduction'] = {'aggregation' : 'sum'}
        g = myderiv(self.x).set_index('edge').value
        u, v = f.align(g)
        assert_allclose(u.values, v.values)

    def test_sum_over_edges(self):
        f = self.f.groupby('site').value.sum()
        self.x['edge_reduction'] = {'aggregation' : 'sum'}
        g = myderiv(self.x).set_index('site').value
        u, v = f.align(g)
        assert_allclose(u.values, v.values)

    def test_avg_over_sites(self):
        f = self.f.groupby('edge').value.mean()
        self.x['site_reduction'] = {'aggregation' : 'avg'}
        g = myderiv(self.x).set_index('edge').value
        u, v = f.align(g)
        assert_allclose(u.values, v.values)

    def test_avg_over_edges(self):
        f = self.f.groupby('site').value.mean()
        self.x['edge_reduction'] = {'aggregation' : 'avg'}
        g = myderiv(self.x).set_index('site').value
        u, v = f.align(g)
        assert_allclose(u.values, v.values)

    def test_edge_sum_site_avg(self):
        f = self.f.groupby('site').value.sum().mean()
        self.x['edge_reduction'] = {'aggregation' : 'sum'}
        self.x['site_reduction'] = {'aggregation' : 'avg'}
        g = myderiv(self.x).value[0]
        assert_allclose(g, f)

    def test_edge_avg_site_sum(self):
        self.x['edge_reduction'] = {'aggregation' : 'avg'}
        self.x['site_reduction'] = {'aggregation' : 'sum'}
        g = myderiv(self.x).value[0]
        f = self.f.groupby('site').value.mean().sum()
        assert_allclose(g, f)

    def test_edge_sum_site_sum(self):
        self.x['edge_reduction'] = {'aggregation' : 'sum'}
        self.x['site_reduction'] = {'aggregation' : 'sum'}
        g = myderiv(self.x).value[0]
        f = self.f.groupby('site').value.sum().sum()
        assert_allclose(g, f)

    def test_edge_avg_site_avg(self):
        f = self.f.groupby('site').value.mean().mean()
        self.x['edge_reduction'] = {'aggregation' : 'avg'}
        self.x['site_reduction'] = {'aggregation' : 'avg'}
        g = myderiv(self.x).value[0]
        assert_allclose(g, f)

    def test_edge_weighted_sum_vs_sum(self):
        f = self.f.groupby('site').value.sum()
        self.x['edge_reduction'] = {'aggregation' : [1, 1, 1, 1, 1, 1, 1]}
        g = myderiv(self.x).value
        u, v = f.align(g)
        assert_allclose(u, v)

    def test_edge_weighted_sum_vs_avg(self):
        p = 1 / 7
        f = self.f.groupby('site').value.mean()
        self.x['edge_reduction'] = {'aggregation' : [p, p, p, p, p, p, p]}
        g = myderiv(self.x).value
        u, v = f.align(g)
        assert_allclose(u, v)

    def test_site_weighted_sum_vs_sum(self):
        f = self.f.groupby('edge').value.sum()
        self.x['site_reduction'] = {'aggregation' : [1, 1]}
        g = myderiv(self.x).value
        u, v = f.align(g)
        assert_allclose(u, v)

    def test_edge_weighted_sum_vs_avg(self):
        p = 1 / 2
        f = self.f.groupby('edge').value.mean()
        self.x['site_reduction'] = {'aggregation' : [p, p]}
        g = myderiv(self.x).value
        u, v = f.align(g)
        assert_allclose(u, v)


class CheckDerivSelection(CheckDeriv, TestCase):

    def __init__(self, *args, **kwargs):
        TestCase.__init__(self, *args, **kwargs)
        CheckDeriv.__init__(self)

    def test_edge_selection(self):
        f = self.f.loc[self.f['edge'].isin([2, 3, 4])]
        f = f.set_index(['site', 'edge']).value
        self.x['edge_reduction'] = {'selection' : [2, 4, 3]}
        g = myderiv(self.x).set_index(['site', 'edge']).value
        u, v = f.align(g)
        assert_allclose(u.values, v.values)

    def test_site_selection(self):
        # Use pandas to select a subset of the results
        # returned from the full arbplf calculations.
        f = self.f.loc[self.f['site'].isin([1])]
        f = f.set_index(['site', 'edge']).value

        # Select using the internal mechanism of the arbplf library,
        # avoiding calculation at unselected sites.
        self.x['site_reduction'] = {'selection' : [1]}
        g = myderiv(self.x).set_index(['site', 'edge']).value

        # Check that the results are equal.
        u, v = f.align(g)
        assert_allclose(u.values, v.values)


class CheckDerivSelectionAggregation(CheckDeriv, TestCase):

    def __init__(self, *args, **kwargs):
        TestCase.__init__(self, *args, **kwargs)
        CheckDeriv.__init__(self)

    def test_site_selection_aggregation_sum(self):
        self.x['site_reduction'] = {
                'selection' : [1, 0, 1],
                'aggregation' : 'sum'}
        self.y['site_reduction'] = {
                'aggregation' : [1, 2]}
        f = myderiv(self.x).set_index('edge').value
        g = myderiv(self.y).set_index('edge').value
        u, v = f.align(g)
        assert_allclose(g, f)

    def test_edge_selection_aggregation_sum(self):
        self.x['edge_reduction'] = {
                'selection' : [2, 4, 6],
                'aggregation' : 'sum'}
        self.y['edge_reduction'] = {
                'aggregation' : [0, 0, 1, 0, 1, 0, 1]}
        f = myderiv(self.x).set_index('site').value
        g = myderiv(self.y).set_index('site').value
        u, v = f.align(g)
        assert_allclose(g, f)

    def test_edge_selection_aggregation_duplicate_sum(self):
        self.x['edge_reduction'] = {
                'selection' : [4, 4],
                'aggregation' : 'sum'}
        self.y['edge_reduction'] = {
                'aggregation' : [0, 0, 0, 0, 2, 0, 0]}
        f = myderiv(self.x).set_index('site').value
        g = myderiv(self.y).set_index('site').value
        u, v = f.align(g)
        assert_allclose(g, f)

    def test_edge_selection_aggregation_cancelling_sum(self):
        self.x['edge_reduction'] = {
                'selection' : [0, 0],
                'aggregation' : [1, -1]}
        self.y['edge_reduction'] = {
                'aggregation' : [0, 0, 0, 0, 0, 0, 0]}
        f = myderiv(self.x).set_index('site').value
        g = myderiv(self.y).set_index('site').value
        u, v = f.align(g)
        assert_allclose(g, f)

    def test_edge_selection_aggregation_weighted_sum(self):
        self.x['edge_reduction'] = {
                'selection' : [2, 6, 4],
                'aggregation' : [3.14, 42, 5]}
        self.y['edge_reduction'] = {
                'aggregation' : [0, 0, 3.14, 0, 5, 0, 42]}
        f = myderiv(self.x).set_index('site').value
        g = myderiv(self.y).set_index('site').value
        u, v = f.align(g)
        assert_allclose(g, f)

"""
