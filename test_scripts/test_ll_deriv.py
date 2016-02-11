"""
Test log likelihoods and derivatives of log likelihoods.

It is only checked for internal consistency, not for accuracy.

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


good_input = {
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
    myll(good_input)

def test_default_deriv():
    myderiv(good_input)

def test_default_ll_vs_empty_reductions():
    f = myll
    x = copy.deepcopy(good_input)
    x['site_reduction'] = {}
    u = f(good_input)
    v = f(x)
    assert_allclose(u.values, v.values)

def test_default_deriv_vs_empty_reductions():
    f = myderiv
    x = copy.deepcopy(good_input)
    x['site_reduction'] = {}
    x['edge_reduction'] = {}
    v = f(good_input)
    u = f(x)
    assert_allclose(u.values, v.values)

def test_finite_differences():
    site = 1
    edge = 2
    delta = 1e-8
    dy = copy.deepcopy(good_input)
    dy['model_and_data']['edge_rate_coefficients'][edge] += delta
    x = myll(good_input).set_index('site').value[site]
    y = myll(dy).set_index('site').value[site]
    z = myderiv(good_input).set_index(['site', 'edge']).value[site, edge]
    #print(x, y)
    #print((y - x) / delta)
    #print(z)
