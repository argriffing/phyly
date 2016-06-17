"""
Test log likelihoods for internal consistency.

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

def _pd_assert_allclose(a, b):
    u, v = a.align(b)
    assert_allclose(u.values, v.values)

def myll(d):
    """
    Provides a dict -> pandas.DataFrame wrapper of the pure JSON arbplf_ll.
    """
    s = arbplf_ll(json.dumps(d))
    df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    return df

def test_default_ll_vs_empty_reductions():
    x = copy.deepcopy(default_in)
    x['site_reduction'] = {}
    u = myll(default_in).set_index('site')
    v = myll(x).set_index('site')
    _pd_assert_allclose(u, v)

def test_selection_order():
    x = copy.deepcopy(default_in)
    x['site_reduction'] = dict(selection=[0, 1])
    y = copy.deepcopy(default_in)
    y['site_reduction'] = dict(selection=[1, 0])
    u = myll(x).set_index('site')
    v = myll(y).set_index('site')
    _pd_assert_allclose(u, v)

def test_sum_vs_only():
    for site in 0, 1:
        x = copy.deepcopy(default_in)
        x['site_reduction'] = dict(selection=[site], aggregation='sum')
        y = copy.deepcopy(default_in)
        y['site_reduction'] = dict(selection=[site], aggregation='only')
        assert_allclose(myll(x), myll(y))

def test_avg_vs_only():
    for site in 0, 1:
        x = copy.deepcopy(default_in)
        x['site_reduction'] = dict(selection=[site], aggregation='avg')
        y = copy.deepcopy(default_in)
        y['site_reduction'] = dict(selection=[site], aggregation='only')
        assert_allclose(myll(x), myll(y))
