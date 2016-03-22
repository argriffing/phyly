"""
Test accuracy using finite differences.

"""
from __future__ import print_function, division

from StringIO import StringIO
import json
import copy
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
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


def myll(d):
    s = arbplf_ll(json.dumps(d))
    df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    return df

def myderiv(d):
    s = arbplf_deriv(json.dumps(d))
    df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    return df

def test_using_finite_differences():
    d_in = default_in

    def my_objective(X):
        d = copy.deepcopy(d_in)
        d['model_and_data']['edge_rate_coefficients'] = X
        d['site_reduction'] = {'aggregation' : 'sum'}
        y = myll(d).value[0]
        return -y

    def my_gradient(X):
        d = copy.deepcopy(d_in)
        d['model_and_data']['edge_rate_coefficients'] = X
        d['site_reduction'] = {'aggregation' : 'sum'}
        y = myderiv(d).set_index('edge').value.values
        return (-y).tolist()

    x0 = d_in['model_and_data']['edge_rate_coefficients']

    y0 = my_objective(x0)
    d0 = my_gradient(x0)

    # check the gradient using numerical differences of 1e-8
    numgrad = []
    delta = 1e-8
    for i in range(len(x0)):
        u = x0[:]
        u[i] += delta
        y = my_objective(u)
        numgrad.append((y - y0) / delta)

    # Require that the finite differences gradient
    # is in the ballpark of the internally computed gradient.
    assert_allclose(numgrad, d0, rtol=1e-3)
