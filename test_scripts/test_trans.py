"""
Test transition count expectations.

It is only checked for internal consistency, not for accuracy.
This test module was written using test_dwell.py as a template.

"""
from __future__ import print_function, division

from StringIO import StringIO
import json
import copy

import numpy as np
import pandas as pd
from numpy.testing import assert_, assert_equal, assert_allclose, TestCase

from arbplf import arbplf_ll, arbplf_trans


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


def mytrans(d):
    """
    Provides a dict -> pandas.DataFrame wrapper of the pure JSON arbplf_trans.
    """
    s = arbplf_trans(json.dumps(d))
    df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    return df

def myll(d):
    """
    Provides a dict -> pandas.DataFrame wrapper of the pure JSON arbplf_ll.
    """
    s = arbplf_ll(json.dumps(d))
    df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    return df

class TestTransEdgeOrder(TestCase):

    def test_arbitrary_edge_permutation(self):
        # Check that permute(f(x)) = f(permute(x))
        # where the permutation is over edges.
        edge_count = len(default_in['model_and_data']['edges'])
        perm = [2, 3, 4, 0, 1, 5, 6]
        trans_reduction = {'selection' : [
            [0, 1], [0, 2], [0, 3],
            [1, 0], [1, 2], [1, 3],
            [2, 0], [2, 1], [2, 3],
            [3, 0], [3, 1], [3, 2]]}
        #
        y = copy.deepcopy(default_in)
        y['site_reduction'] = {'aggregation' : 'sum'}
        y['trans_reduction'] = trans_reduction
        f = mytrans(y).set_index('edge').value
        fperm = copy.deepcopy(f)
        for u, v in enumerate(perm):
            fperm[u] = f[v]
        #
        x = copy.deepcopy(default_in)
        x['site_reduction'] = {'aggregation' : 'sum'}
        x['trans_reduction'] = trans_reduction
        #
        xm = x['model_and_data']
        ym = y['model_and_data']
        for u, v in enumerate(perm):
            xm['edges'][u] = copy.deepcopy(
                    ym['edges'][v])
            xm['edge_rate_coefficients'][u] = copy.deepcopy(
                    ym['edge_rate_coefficients'][v])
        g = mytrans(x).set_index('edge').value
        #
        _assert_allclose_series(fperm, g)

    def test_rate_mixture_state_aggregation(self):
        rates = [2, 3]
        prior = [0.25, 0.75]
        trans_selection = [[0, 1], [3, 1], [2, 3]]
        trans_weights = [0.1, 4.2, 0.3]
        trans_reduction = {
                'selection' : trans_selection,
                'aggregation' : trans_weights}
        rate_mixture = dict(rates=rates, prior=prior)
        # first likelihood and trans
        r = rates[0]
        p = prior[0]
        x = copy.deepcopy(default_in)
        xm = x['model_and_data']
        xm['edge_rate_coefficients'] = [
                u*r for u in xm['edge_rate_coefficients']]
        ll = myll(x).set_index('site').value
        x['edge_reduction'] = {'aggregation' : 'sum'}
        x['trans_reduction'] = trans_reduction
        trans = mytrans(x).set_index('site').value
        first_lhood = p * np.exp(ll)
        first_trans = trans
        # second lhood and trans
        r = rates[1]
        p = prior[1]
        x = copy.deepcopy(default_in)
        xm = x['model_and_data']
        xm['edge_rate_coefficients'] = [
                u*r for u in xm['edge_rate_coefficients']]
        ll = myll(x).set_index('site').value
        x['edge_reduction'] = {'aggregation' : 'sum'}
        x['trans_reduction'] = trans_reduction
        trans = mytrans(x).set_index('site').value
        second_lhood = p * np.exp(ll)
        second_trans = trans
        # combine the separately calculated values
        a = first_lhood * first_trans
        b = second_lhood * second_trans
        c = first_lhood + second_lhood
        desired = (a + b) / c
        # compute the trans expectations of the mixture in a single call
        x = copy.deepcopy(default_in)
        xm = x['model_and_data']
        xm['rate_mixture'] = rate_mixture
        x['edge_reduction'] = {'aggregation' : 'sum'}
        x['trans_reduction'] = trans_reduction
        actual = mytrans(x).set_index('site').value
        # check that the two calculations are equivalent
        _assert_allclose_series(actual, desired)

    def test_rate_mixture_no_state_aggregation(self):
        # FIXME .values may expect a particular ordering of the indices
        rates = [2, 3]
        prior = [0.25, 0.75]
        # FIXME test with unsorted trans_selection
        trans_selection = [[0, 1], [2, 3], [3, 1]]
        trans_weights = [0.1, 4.2, 0.3]
        rate_mixture = dict(rates=rates, prior=prior)
        # first likelihood and trans
        r = rates[0]
        p = prior[0]
        x = copy.deepcopy(default_in)
        xm = x['model_and_data']
        xm['edge_rate_coefficients'] = [
                u*r for u in xm['edge_rate_coefficients']]
        out = myll(x)
        ll = out.set_index('site').value.values
        x['edge_reduction'] = {'aggregation' : 'sum'}
        x['trans_reduction'] = {'selection' : trans_selection}
        out = mytrans(x)
        piv = pd.pivot_table(out,
                index=['first_state', 'second_state'],
                columns=['site'],
                values='value')
        first_trans = piv.values
        first_lhood = p * np.exp(ll)
        # second lhood and trans
        r = rates[1]
        p = prior[1]
        x = copy.deepcopy(default_in)
        xm = x['model_and_data']
        xm['edge_rate_coefficients'] = [
                u*r for u in xm['edge_rate_coefficients']]
        out = myll(x)
        ll = out.set_index('site').value.values
        x['edge_reduction'] = {'aggregation' : 'sum'}
        x['trans_reduction'] = {'selection' : trans_selection}
        out = mytrans(x)
        piv = pd.pivot_table(out,
                index=['first_state', 'second_state'],
                columns=['site'],
                values='value')
        #second_trans = mytrans(x).pivot('state', 'site', 'value').values
        second_trans = piv.values
        second_lhood = p * np.exp(ll)
        # combine the separately calculated values
        a = first_lhood * first_trans
        b = second_lhood * second_trans
        c = first_lhood + second_lhood
        desired = np.dot(trans_weights, (a + b) / c)
        # compute the trans expectations of the mixture in a single call
        x = copy.deepcopy(default_in)
        xm = x['model_and_data']
        xm['rate_mixture'] = rate_mixture
        x['trans_reduction'] = {
                'selection' : trans_selection,
                'aggregation' : trans_weights}
        x['edge_reduction'] = {'aggregation' : 'sum'}
        out = mytrans(x)
        actual = out.set_index('site').value.values
        # check that the two calculations are equivalent
        assert_allclose(actual, desired)
