"""
Test dwell expectations.

It is only checked for internal consistency, not for accuracy.

"""
from __future__ import print_function, division

from StringIO import StringIO
import json
import copy

import numpy as np
import pandas as pd
from numpy.testing import assert_, assert_equal, assert_allclose, TestCase

from arbplf import arbplf_ll, arbplf_dwell


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


def mydwell(d):
    """
    Provides a dict -> pandas.DataFrame wrapper of the pure JSON arbplf_dwell.
    """
    s = arbplf_dwell(json.dumps(d))
    df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    return df

def myll(d):
    """
    Provides a dict -> pandas.DataFrame wrapper of the pure JSON arbplf_ll.
    """
    s = arbplf_ll(json.dumps(d))
    df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    return df

class TestDwellEdgeOrder(TestCase):

    def test_arbitrary_edge_permutation(self):
        # Check that permute(f(x)) = f(permute(x))
        # where the permutation is over edges.
        edge_count = len(default_in['model_and_data']['edges'])
        perm = [2, 3, 4, 0, 1, 5, 6]
        #
        y = copy.deepcopy(default_in)
        y['site_reduction'] = {'aggregation' : 'sum'}
        f = mydwell(y).set_index('edge').value
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
        g = mydwell(x).set_index('edge').value
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
        y['state_reduction'] = {'aggregation' : 'sum'}
        f = mydwell(y).set_index('edge').value
        fperm = copy.deepcopy(f)
        for u, v in enumerate(perm):
            fperm[u] = f[v]
        fsel = fperm[selection]
        #
        x = copy.deepcopy(default_in)
        x['site_reduction'] = {'aggregation' : 'sum'}
        x['state_reduction'] = {'aggregation' : 'sum'}
        x['edge_reduction'] = {'selection' : selection}
        #
        xm = x['model_and_data']
        ym = y['model_and_data']
        for u, v in enumerate(perm):
            xm['edges'][u] = copy.deepcopy(
                    ym['edges'][v])
            xm['edge_rate_coefficients'][u] = copy.deepcopy(
                    ym['edge_rate_coefficients'][v])
        g = mydwell(x).set_index('edge').value
        #
        _assert_allclose_series(fsel, g)

    def test_rate_mixture(self):
        rates = [2, 3]
        prior = [0.25, 0.75]
        weights = [0.1, 0.2, 0.8, 2.0]
        rate_mixture = dict(rates=rates, prior=prior)
        # first likelihood and dwell
        r = rates[0]
        p = prior[0]
        x = copy.deepcopy(default_in)
        xm = x['model_and_data']
        xm['edge_rate_coefficients'] = [
                u*r for u in xm['edge_rate_coefficients']]
        ll = myll(x).set_index('site').value
        x['edge_reduction'] = {'aggregation' : 'sum'}
        x['state_reduction'] = {'aggregation' : weights}
        dwell = mydwell(x).set_index('site').value
        first_lhood = p * np.exp(ll)
        first_dwell = dwell
        # second lhood and dwell
        r = rates[1]
        p = prior[1]
        x = copy.deepcopy(default_in)
        xm = x['model_and_data']
        xm['edge_rate_coefficients'] = [
                u*r for u in xm['edge_rate_coefficients']]
        ll = myll(x).set_index('site').value
        x['edge_reduction'] = {'aggregation' : 'sum'}
        x['state_reduction'] = {'aggregation' : weights}
        dwell = mydwell(x).set_index('site').value
        second_lhood = p * np.exp(ll)
        second_dwell = dwell
        # combine the separately calculated values
        a = first_lhood * first_dwell
        b = second_lhood * second_dwell
        c = first_lhood + second_lhood
        desired = (a + b) / c
        # compute the dwell expectations of the mixture in a single call
        x = copy.deepcopy(default_in)
        xm = x['model_and_data']
        xm['rate_mixture'] = rate_mixture
        x['edge_reduction'] = {'aggregation' : 'sum'}
        x['state_reduction'] = {'aggregation' : weights}
        actual = mydwell(x).set_index('site').value
        # check that the two calculations are equivalent
        _assert_allclose_series(actual, desired)
