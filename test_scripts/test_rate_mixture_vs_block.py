"""
Test some rate mixture implementations vs. equivalent block rate matrix models.

This is particularly easy to test for functions whose output does
not include per-state summaries or summaries that are aggregated over states.
 - ll
 - deriv
 - hessian-based functions
 - edge rate EM update
so, basically all functions except dwell and trans expectations
and conditional marginal distributions at nodes.

"""
from __future__ import print_function, division

from StringIO import StringIO
from functools import partial
import json
import copy
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from numpy.testing import (
        assert_, assert_equal, assert_raises, assert_allclose, TestCase)

from arbplf import (
        arbplf_ll, arbplf_marginal,
        arbplf_dwell, arbplf_trans, arbplf_em_update,
        arbplf_newton_delta, arbplf_newton_update,
        arbplf_newton_refine, arbplf_deriv, arbplf_hess,
        arbplf_inv_hess,
        )

# These are easy to test, so they are tested in this module.
# For now I'm skipping the newton refinement.
_easy_funcs = (
        arbplf_ll,
        arbplf_deriv,
        arbplf_hess, arbplf_inv_hess,
        arbplf_newton_delta, arbplf_newton_update,
        arbplf_em_update)

# These are still easy to test, but are a bit more difficult.
# They are not tested in this module.
_less_easy_funcs = (
        arbplf_marginal, arbplf_dwell, arbplf_trans, arbplf_newton_refine)

def _df(f, d):
    s_in = json.dumps(d)
    s_out = f(s_in)
    df = pd.read_json(StringIO(s_out), orient='split', precise_float=True)
    return df

def _assert_allclose_series(a, b):
    # Align the layouts of the two pandas Series objects,
    # preparing to compare their values.
    # The align member function returns two new pandas Series objects.
    u, v = a.align(b)
    assert_allclose(u.values, v.values)


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
 "site_reduction" : {"aggregation" : "sum"},
}


def test_rate_mixture_equivalence():
    # Compare rates across sites vs. an equivalent block diagonal formulation.
    rates = [1, 2]
    prior = [0.25, 0.75]
    rate_mixture = dict(rates=rates, prior=prior)
    #
    for f in _easy_funcs:
        print(f.__name__)
        # run the analysis using a block diagonal rate matrix
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
        u = _df(f, x)
        # run the analysis using a rate mixture
        x = copy.deepcopy(default_in)
        x['model_and_data']['rate_mixture'] = rate_mixture
        v = _df(f, x)
        # check that the two analyses give the same results
        assert_allclose(u.values, v.values)

def test_arbitrary_edge_permutation():
    # Check that permute(f(x)) = f(permute(x))
    # where the permutation is over edges.
    rates = [1, 2]
    prior = [0.25, 0.75]
    rate_mixture = dict(rates=rates, prior=prior)
    #
    edge_count = len(default_in['model_and_data']['edges'])
    perm = [2, 3, 4, 0, 1, 5, 6]
    # These functions have an 'edge' output column,
    # but they do not have a 'state' output column.
    for f in arbplf_deriv, arbplf_em_update:
        print(f.__name__)
        #
        y = copy.deepcopy(default_in)
        y['model_and_data']['rate_mixture'] = rate_mixture
        u = _df(f, y).set_index('edge').value
        uperm = copy.deepcopy(u)
        for a, b in enumerate(perm):
            uperm[a] = u[b]
        #
        x = copy.deepcopy(default_in)
        x['model_and_data']['rate_mixture'] = rate_mixture
        #
        xm = x['model_and_data']
        ym = y['model_and_data']
        for a, b in enumerate(perm):
            xm['edges'][a] = copy.deepcopy(
                    ym['edges'][b])
            xm['edge_rate_coefficients'][a] = copy.deepcopy(
                    ym['edge_rate_coefficients'][b])
        v = _df(f, x).set_index('edge').value
        #
        _assert_allclose_series(uperm, v)
