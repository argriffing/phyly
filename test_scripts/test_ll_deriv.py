"""
Test log likelihoods and derivatives of log likelihoods.

It is only checked for internal consistency, not for accuracy.

"""
from __future__ import print_function, division

from StringIO import StringIO
import json
import copy
import random

import numpy as np
import pandas as pd
from numpy.testing import (
        assert_, assert_equal, assert_raises, assert_allclose, TestCase)

from arbplf import arbplf_ll
from arbplf import arbplf_deriv


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
    Provides a dict -> pandas.DataFrame wrapper of the pure JSON arbplf_ll.
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

def test_default_ll():
    myll(default_in)

def test_default_deriv():
    myderiv(default_in)

def test_default_ll_vs_empty_reductions():
    f = myll
    x = copy.deepcopy(default_in)
    x['site_reduction'] = {}
    u = f(default_in)
    v = f(x)
    assert_allclose(u.values, v.values)

def test_default_deriv_vs_empty_reductions():
    f = myderiv
    x = copy.deepcopy(default_in)
    x['site_reduction'] = {}
    x['edge_reduction'] = {}
    v = f(default_in)
    u = f(x)
    assert_allclose(u.values, v.values)


def test_deriv_sum_over_sites():

    # Create a full pandas DataFrame with columns 'site', 'edge', and 'value'.
    # The 'site' and 'edge' columns contain indices,
    # and the 'value' column contains floating point derivatives of the
    # phylogenetic log likelihood.
    full = myderiv(default_in)

    # Reduce the table to a pandas Series, where entries are indexed
    # by edge and whose values are floating point derivatives.
    # Reduction is by summation.
    #pd_sum = pd.pivot_table(full, 'value', columns='edge', aggfunc=np.sum)

    # Alternatively, create a pandas Series.
    # The values are summed over all axes except 'edge'.
    pd_sum = full.groupby('edge').value.sum()

    # Recompute the phylogenetic log likelihood derivatives,
    # but this time aggregating sites by summation, using internal
    # mechanisms of the arbplf_deriv C function that make use of
    # linearity of summation and of error bounds.
    # The result is a pandas Series.
    # This method is potentially more accurate and efficient.
    # Edges are selected in a weird order for the purpose of illustrating
    # layout differences.
    x = copy.deepcopy(default_in)
    x['site_reduction'] = {'aggregation' : 'sum'}
    x['edge_reduction'] = {'selection' : [4, 5, 6, 3, 2, 1, 0]}
    phyly_sum = myderiv(x).set_index('edge').value

    # These pandas Series objects should contain the same information,
    # up to small numerical errors, but the underlying arrays are not
    # directly comparable because the internal layouts are different.
    assert_(not np.allclose(pd_sum.values, phyly_sum.values))

    _assert_allclose_series(pd_sum, phyly_sum)


class CheckDeriv(object):

    def __init__(self):
        self.f = myderiv(default_in)

    def setUp(self):
        self.x = copy.deepcopy(default_in)
        self.y = copy.deepcopy(default_in)


class TestDerivAggregation(CheckDeriv, TestCase):

    def __init__(self, *args, **kwargs):
        TestCase.__init__(self, *args, **kwargs)
        CheckDeriv.__init__(self)

    def test_sum_over_sites(self):
        f = self.f.groupby('edge').value.sum()
        self.x['site_reduction'] = {'aggregation' : 'sum'}
        g = myderiv(self.x).set_index('edge').value
        _assert_allclose_series(f, g)

    def test_sum_over_edges(self):
        f = self.f.groupby('site').value.sum()
        self.x['edge_reduction'] = {'aggregation' : 'sum'}
        g = myderiv(self.x).set_index('site').value
        _assert_allclose_series(f, g)

    def test_avg_over_sites(self):
        f = self.f.groupby('edge').value.mean()
        self.x['site_reduction'] = {'aggregation' : 'avg'}
        g = myderiv(self.x).set_index('edge').value
        _assert_allclose_series(f, g)

    def test_avg_over_edges(self):
        f = self.f.groupby('site').value.mean()
        self.x['edge_reduction'] = {'aggregation' : 'avg'}
        g = myderiv(self.x).set_index('site').value
        _assert_allclose_series(f, g)

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
        _assert_allclose_series(f, g)

    def test_edge_weighted_sum_vs_avg(self):
        p = 1 / 7
        f = self.f.groupby('site').value.mean()
        self.x['edge_reduction'] = {'aggregation' : [p, p, p, p, p, p, p]}
        g = myderiv(self.x).value
        _assert_allclose_series(f, g)

    def test_site_weighted_sum_vs_sum(self):
        f = self.f.groupby('edge').value.sum()
        self.x['site_reduction'] = {'aggregation' : [1, 1]}
        g = myderiv(self.x).value
        _assert_allclose_series(f, g)

    def test_edge_weighted_sum_vs_avg(self):
        p = 1 / 2
        f = self.f.groupby('edge').value.mean()
        self.x['site_reduction'] = {'aggregation' : [p, p]}
        g = myderiv(self.x).value
        _assert_allclose_series(f, g)


class TestDerivSelection(CheckDeriv, TestCase):

    def __init__(self, *args, **kwargs):
        TestCase.__init__(self, *args, **kwargs)
        CheckDeriv.__init__(self)

    def test_edge_selection(self):
        f = self.f.loc[self.f['edge'].isin([2, 3, 4])]
        f = f.set_index(['site', 'edge']).value
        self.x['edge_reduction'] = {'selection' : [2, 4, 3]}
        g = myderiv(self.x).set_index(['site', 'edge']).value
        _assert_allclose_series(f, g)

    def test_site_selection(self):
        # Use pandas to select a subset of the results
        # returned from the full arbplf calculations.
        f = self.f.loc[self.f['site'].isin([1])]
        f = f.set_index(['site', 'edge']).value

        # Select using the internal mechanism of the arbplf library,
        # avoiding calculation at unselected sites.
        self.x['site_reduction'] = {'selection' : [1]}
        g = myderiv(self.x).set_index(['site', 'edge']).value

        _assert_allclose_series(f, g)


class TestDerivSelectionAggregation(CheckDeriv, TestCase):

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
        _assert_allclose_series(f, g)

    def test_edge_selection_aggregation_sum(self):
        self.x['edge_reduction'] = {
                'selection' : [2, 4, 6],
                'aggregation' : 'sum'}
        self.y['edge_reduction'] = {
                'aggregation' : [0, 0, 1, 0, 1, 0, 1]}
        f = myderiv(self.x).set_index('site').value
        g = myderiv(self.y).set_index('site').value
        _assert_allclose_series(f, g)

    def test_edge_selection_aggregation_duplicate_sum(self):
        self.x['edge_reduction'] = {
                'selection' : [4, 4],
                'aggregation' : 'sum'}
        self.y['edge_reduction'] = {
                'aggregation' : [0, 0, 0, 0, 2, 0, 0]}
        f = myderiv(self.x).set_index('site').value
        g = myderiv(self.y).set_index('site').value
        _assert_allclose_series(f, g)

    def test_edge_selection_aggregation_cancelling_sum(self):
        self.x['edge_reduction'] = {
                'selection' : [0, 0],
                'aggregation' : [1, -1]}
        self.y['edge_reduction'] = {
                'aggregation' : [0, 0, 0, 0, 0, 0, 0]}
        f = myderiv(self.x).set_index('site').value
        g = myderiv(self.y).set_index('site').value
        _assert_allclose_series(f, g)

    def test_edge_selection_aggregation_weighted_sum(self):
        self.x['edge_reduction'] = {
                'selection' : [2, 6, 4],
                'aggregation' : [3.14, 42, 5]}
        self.y['edge_reduction'] = {
                'aggregation' : [0, 0, 3.14, 0, 5, 0, 42]}
        f = myderiv(self.x).set_index('site').value
        g = myderiv(self.y).set_index('site').value
        _assert_allclose_series(f, g)


class TestDerivEdgeOrder(TestCase):

    def test_arbitrary_edge_permutation(self):
        # Check that permute(f(x)) = f(permute(x))
        # where the permutation is over edges.
        edge_count = len(default_in['model_and_data']['edges'])
        perm = [2, 3, 4, 0, 1, 5, 6]
        #
        y = copy.deepcopy(default_in)
        y['site_reduction'] = {'aggregation' : 'sum'}
        f = myderiv(y).set_index('edge').value
        fperm = copy.deepcopy(f)
        for u, v in enumerate(perm):
            fperm[u] = f[v]
        #
        x = copy.deepcopy(default_in)
        x['site_reduction'] = {'aggregation' : 'sum'}
        #
        xm = x['model_and_data']
        ym = y['model_and_data']
        for u, v in enumerate(perm):
            xm['edges'][u] = copy.deepcopy(
                    ym['edges'][v])
            xm['edge_rate_coefficients'][u] = copy.deepcopy(
                    ym['edge_rate_coefficients'][v])
        g = myderiv(x).set_index('edge').value
        #
        _assert_allclose_series(fperm, g)

    def test_selection_with_arbitrary_edge_permutation(self):
        # Check that select(permute(f(x))) = select(f(permute(x)))
        # where the permutation is over edges.
        edge_count = len(default_in['model_and_data']['edges'])
        perm = [2, 3, 4, 0, 1, 5, 6]
        selection = [1, 1, 2, 6, 5]
        #
        y = copy.deepcopy(default_in)
        y['site_reduction'] = {'aggregation' : 'sum'}
        f = myderiv(y).set_index('edge').value
        fperm = copy.deepcopy(f)
        for u, v in enumerate(perm):
            fperm[u] = f[v]
        fsel = fperm[selection]
        #
        x = copy.deepcopy(default_in)
        x['site_reduction'] = {'aggregation' : 'sum'}
        x['edge_reduction'] = {'selection' : selection}
        #
        xm = x['model_and_data']
        ym = y['model_and_data']
        for u, v in enumerate(perm):
            xm['edges'][u] = copy.deepcopy(
                    ym['edges'][v])
            xm['edge_rate_coefficients'][u] = copy.deepcopy(
                    ym['edge_rate_coefficients'][v])
        g = myderiv(x).set_index('edge').value
        #
        _assert_allclose_series(fsel, g)
