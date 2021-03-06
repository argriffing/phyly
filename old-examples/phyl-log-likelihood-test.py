"""
"""
from __future__ import print_function, division

from StringIO import StringIO
from functools import partial
import json
import itertools
import math

import numpy as np
import pandas as pd
import scipy.stats
from scipy import optimize
from scipy.special import gamma, gammainc, expit, logit

from arbplf import arbplf_ll, arbplf_deriv

_edges = [[4, 0], [4, 1], [5, 4], [5, 2], [5, 3]]
_character_definitions = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 1, 1]]
_character_data = [
        [0, 2, 1, 0, 4, 4],
        [0, 0, 3, 0, 4, 4],
        [0, 1, 1, 0, 4, 4],
        [3, 3, 3, 3, 4, 4],
        [2, 2, 2, 2, 4, 4],
        [2, 2, 2, 2, 4, 4],
        [1, 0, 0, 1, 4, 4],
        [3, 3, 3, 2, 4, 4],
        [2, 1, 2, 2, 4, 4],
        [3, 3, 3, 3, 4, 4],
        [2, 2, 2, 2, 4, 4],
        [1, 1, 1, 1, 4, 4],
        [0, 0, 0, 2, 4, 4],
        [1, 1, 1, 1, 4, 4],
        [2, 2, 2, 1, 4, 4],
        [3, 3, 3, 3, 4, 4],
        [1, 1, 2, 0, 4, 4]]

def _finite_diffs(objective, request, D, X, y):
    for i, b in enumerate(request):
        if not b:
            continue
        U = X.copy()
        if X[i]:
            h = X[i] * 1e-8
        else:
            h = 1e-8
        U[i] = X[i] + h
        v = objective(U)
        D[i] = (v - y) / h


def T92(k, t=0.5):
    # try to use the Bio++ T92 parameterization
    # http://biopp.univ-montp2.fr/Documents/ClassDocumentation/bpp-phyl/html/classbpp_1_1T92.html#details
    R = [[0, 1, k, 1],
         [1, 0, 1, k],
         [k, 1, 0, 1],
         [1, k, 1, 0]]
    pi = [(1-t)/2, t/2, t/2, (1-t)/2]
    for j, p in enumerate(pi):
        for i in range(4):
            R[i][j] *= p
    x = 0
    for i, p in enumerate(pi):
        r = 0
        for j in range(4):
            if i != j:
                r += R[i][j]
        x += r * p
    return R, x


def block_rate_mixture(R, rates):
    """
    Expand a rate matrix into a block diagonal rate matrix mixture.
    Block i of the mixture is scaled by rates[i].
    Diagonal entries remain undefined.
    """
    N = len(R)
    K = len(rates)
    M = []
    for i in range(N*K):
        M.append([0]*N*K)
    for k in range(K):
        for i in range(N):
            for j in range(N):
                M[k*K + i][k*K + j] = R[i][j] * rates[k]
    return M


def _numerically_discretized_gamma(N, alpha, use_mean=1):
    if use_mean != 1:
        raise NotImplementedException
    def f(x):
        return x
    beta = alpha
    rv = scipy.stats.gamma(alpha, scale=1/beta)
    x = rv.ppf(np.linspace(0, 1, num=N+1))
    pairs = zip(x[:-1], x[1:])
    rates = []
    for lb, ub in pairs:
        rates.append(rv.expect(f, lb=lb, ub=ub, epsrel=1e-14))
    rates = N * np.array(rates)
    #print(rates)
    return rates

def discretized_gamma(N, alpha, use_mean=1):
    if use_mean != 1:
        raise NotImplementedException
    beta = alpha
    scale = 1 / beta
    rv = scipy.stats.gamma(alpha, scale=scale)
    x = rv.ppf(np.linspace(0, 1, num=N+1))
    pairs = zip(x[:-1], x[1:])
    rates = []
    def f(x):
        return gammainc(alpha+1, x/scale)
    #print(alpha)
    for lb, ub in pairs:
        l = f(lb)
        u = f(ub)
        r = u - l
        #print(lb, ub, l, u, r)
        rates.append(r)
    rates = N * np.array(rates)
    #print(rates)
    return rates

def generic_neg_ll(get_s, X):
    s_in = get_s(X)
    s_out = arbplf_ll(s_in)
    df = pd.read_json(StringIO(s_out), orient='split', precise_float=True)
    y = -df.value.values[0]
    print(y)
    return y

def generic_objective(get_s, X):
    s_in = get_s(X)
    neg_ll = partial(generic_neg_ll, get_s)

    # compute negative log likelihood
    s_out = arbplf_ll(s_in)
    df = pd.read_json(StringIO(s_out), orient='split', precise_float=True)
    y = -df.value.values[0]

    # explicitly compute derivatives with respect to edge rate coefficients
    s_out = arbplf_deriv(s_in)
    df = pd.read_json(StringIO(s_out), orient='split', precise_float=True)
    d = -df.set_index('edge').value.values

    edge_count = len(_edges)
    D = np.concatenate((d, [0]*3))
    request = np.array([0]*edge_count + [1]*3)
    _finite_diffs(neg_ll, request, D, X, y)
    print(y, D)
    return y, D

def mixture_get_s(X):
    # same as block_objective, but implemented with a rate mixture
    #print(X)
    kappa, logit_theta, alpha = X[-3:]
    edge_rates = X[:-3].tolist()

    #print('alpha:', alpha)

    theta = expit(logit_theta)
    t = theta
    pi = [(1-t)/2, t/2, t/2, (1-t)/2]
    #print('pi:', pi)

    #alpha = 1.0

    rate_category_count = 4
    state_count = 4
    edge_count = len(_edges)
    node_count = edge_count + 1

    # Use a T92 rate matrix.
    R, x = T92(kappa, theta)
    mixture_rates = discretized_gamma(rate_category_count, alpha)
    #mixture_rates = _numerically_discretized_gamma(rate_category_count, alpha)
    rate_matrix = block_rate_mixture(R, [1])

    # Sequences at internal nodes consist of unobserved nucleotide N.
    # The gamma rate class is unobserved.

    rate_mixture = dict(
            rates = mixture_rates.tolist(),
            prior = 'uniform_distribution')
    gamma_rate_mixture = dict(gamma_shape=alpha, gamma_categories=4)
    model_and_data = {
            "edges" : _edges,
            "rate_divisor" : 100 * x,
            'gamma_rate_mixture' : gamma_rate_mixture,
            "edge_rate_coefficients" : edge_rates,
            "rate_matrix" : rate_matrix,
            'root_prior' : pi,
            'character_definitions' : _character_definitions,
            'character_data' : _character_data,
            }
    d = {
            "model_and_data" : model_and_data,
            "site_reduction" : {"aggregation" : "sum"}}
    s = json.dumps(d)
    return s


def block_get_s(X):
    # implemented with a block matrix
    kappa, logit_theta, alpha = X[-3:]
    edge_rates = X[:-3].tolist()

    theta = expit(logit_theta)
    t = theta
    pi = [(1-t)/2, t/2, t/2, (1-t)/2]

    rate_category_count = 4
    state_count = 4
    edge_count = len(_edges)
    node_count = edge_count + 1

    # Use a T92 rate matrix.
    R, x = T92(kappa, theta)
    mixture_rates = discretized_gamma(rate_category_count, alpha)
    rate_matrix = block_rate_mixture(R, mixture_rates)

    # Sequences at internal nodes consist of unobserved nucleotide N.
    # The gamma rate class is unobserved.
    defs = [lst*4 for lst in _character_definitions]
    pi = [0.25 * x for x in pi*4]

    model_and_data = {
            "edges" : _edges,
            "rate_divisor" : 100 * x,
            "edge_rate_coefficients" : edge_rates,
            "rate_matrix" : rate_matrix,
            'root_prior' : pi,
            'character_definitions' : defs,
            'character_data' : _character_data,
            }
    d = {
            "model_and_data" : model_and_data,
            "site_reduction" : {"aggregation" : "sum"}}
    s = json.dumps(d)
    return s


def main():
    get_s = block_get_s
    #get_s = mixture_get_s

    objective = partial(generic_objective, get_s)
    neg_ll = partial(generic_neg_ll, get_s)

    edge_rates = [1, 30, 1, 30, 30]
    kappa = 0.2
    theta = 0.5
    alpha = 2.0
    #edge_rates = [1, 2, 3, 1, 10]
    #alpha = 1.0
    #kappa = 3.0
    #theta = 0.5
    X0 = np.array(edge_rates + [kappa, logit(theta), alpha], dtype=float)
    print('%.20g' % neg_ll(X0))

    desired_ll = 85.030942031997312824
    #edge_rates = [1, 2, 3, 1, 10]
    #kappa = 3
    X0 = np.array(edge_rates + [kappa, logit(theta), alpha], dtype=float)
    a = 1e-6
    bounds = [(a, None) for i in X0[:5]] + [(a, None), (a, 1), (a, None)]
    result = optimize.minimize(
            objective, X0, method='L-BFGS-B', jac=True, bounds=bounds)
    #result = optimize.minimize(
            #neg_ll, X0, method='L-BFGS-B', bounds=bounds)
    print(result)

main()
