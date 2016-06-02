"""
Uses test_rate_mixture_vs_block.py as a template.

See Figure 1 of
"Maximum Likelihood Phylogenetic Estimation from DNA Sequences
with Variable Rates over Sites: Approximate Methods"
Ziheng Yang (1994) J Mol Evol.
http://abacus.gene.ucl.ac.uk/ziheng/pdf/1994YangJMEv39p306.pdf

"""
from __future__ import print_function, division

from StringIO import StringIO
from functools import partial
import json
import copy
import sys
import time

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
        arbplf_newton_delta, arbplf_newton_update)

# These are still easy to test, but are a bit more difficult.
# They are not tested in this module.
_less_easy_funcs = (
        arbplf_marginal, arbplf_dwell, arbplf_trans,
        arbplf_newton_refine, arbplf_em_update)

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
     "root_prior" : "uniform_distribution",
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
         [1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1]],
         [
         [1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]]},
 "site_reduction" : {"aggregation" : "sum"},
}

def test_rate_mixture_equivalence():
    # Check the gamma discretization.
    gamma_rate_mixture = dict(
            gamma_shape = 0.5,
            gamma_categories = 4)
    rate_mixture = dict(
            rates = [
                0.0333877533835995,
                0.251915917593438,
                0.820268481973649,
                2.89442784704931],
            prior = 'uniform_distribution')
    #
    for f in _easy_funcs:
        print(f.__name__)
        # let the engine do the discretization
        x = copy.deepcopy(default_in)
        x['model_and_data']['gamma_rate_mixture'] = gamma_rate_mixture
        u = _df(f, x)
        # use a precomputed discretization
        x = copy.deepcopy(default_in)
        x['model_and_data']['rate_mixture'] = rate_mixture
        v = _df(f, x)
        # check that the two analyses give the same results
        assert_allclose(u.values, v.values)

def test_invariable_site_rate_mixture_equivalence():
    # Check the gamma discretization.
    gamma_rate_mixture = dict(
            gamma_shape = 0.5,
            gamma_categories = 4,
            invariable_prior = 0.3)
    rate_mixture = dict(
            rates = [
                0.0333877533835995 / 0.7,
                0.251915917593438 / 0.7,
                0.820268481973649 / 0.7,
                2.89442784704931 / 0.7,
                0.0],
            prior = [
                0.7 / 4,
                0.7 / 4,
                0.7 / 4,
                0.7 / 4,
                0.3])
    #
    for f in _easy_funcs:
        print(f.__name__)
        # let the engine do the discretization
        x = copy.deepcopy(default_in)
        x['model_and_data']['gamma_rate_mixture'] = gamma_rate_mixture
        u = _df(f, x)
        # use a precomputed discretization
        x = copy.deepcopy(default_in)
        x['model_and_data']['rate_mixture'] = rate_mixture
        v = _df(f, x)
        # check that the two analyses give the same results
        assert_allclose(u.values, v.values)

def test_small_gamma_shape():
    # Check the gamma discretization.
    # When the shape is near zero,
    # all but one category has nearly zero rate.
    # Because the gamma_rate_mixture is internally normalized to have
    # expected rate 1, in practice this means it has effectively a single
    # category and this category has rate 4 and probability 1/4.
    gamma_rate_mixture = dict(
            gamma_shape = 1e-6,
            gamma_categories = 4)
    rate_mixture = dict(
            rates = [0, 4],
            prior = [3/4, 1/4])
    #
    for f in _easy_funcs:
        print(f.__name__)
        start = time.time()
        # let the engine do the discretization
        x = copy.deepcopy(default_in)
        x['model_and_data']['gamma_rate_mixture'] = gamma_rate_mixture
        u = _df(f, x)
        # use a precomputed discretization
        x = copy.deepcopy(default_in)
        x['model_and_data']['rate_mixture'] = rate_mixture
        v = _df(f, x)
        # check that the two analyses give the same results
        assert_allclose(u.values, v.values)
        print(u)
        print(v)
        print('time:', time.time() - start)
