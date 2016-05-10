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

from arbplf import arbplf_dwell


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
