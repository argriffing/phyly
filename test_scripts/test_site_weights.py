"""
Test an invariant related to site weights.

In particular, when a site reduction is requested in the presence
of redundant sites, only the sum of weights of those sites should matter.

This invariant is tested for all functions except arbplf_newton_refine,
and it is tested in the presence and absence of rate mixtures.

"""
from __future__ import print_function, division

import json
import copy
from numpy.testing import assert_equal


from arbplf import (
        arbplf_ll, arbplf_marginal,
        arbplf_dwell, arbplf_trans, arbplf_em_update,
        arbplf_newton_delta, arbplf_newton_update,
        #arbplf_newton_refine,
        arbplf_deriv, arbplf_hess, arbplf_inv_hess,
        )

model_a = dict(
        edges = [[1, 0]],
        edge_rate_coefficients = [2.0],
        rate_matrix = [
            [0, 1],
            [3, 0]],
        probability_array = [
            [[1, 0], [1, 0]],
            [[0, 1], [1, 0]]],
        )

model_b = dict(
        edges = [[1, 0]],
        edge_rate_coefficients = [2.0],
        rate_matrix = [
            [0, 1],
            [3, 0]],
        probability_array = [
            [[0, 1], [1, 0]],
            [[1, 0], [1, 0]],
            [[0, 1], [1, 0]],
            ],
        )

# properties of site reduction will be tested
site_sum_a = dict(site_reduction=dict(aggregation=[2, 4]))
site_sum_b = dict(site_reduction=dict(aggregation=[1, 2, 3]))

# define additional axis reductions
node_sum = dict(node_reduction=dict(aggregation='sum'))
edge_sum = dict(edge_reduction=dict(aggregation='sum'))
state_sum = dict(state_reduction=dict(aggregation='sum'))
trans_sum = dict(trans_reduction=dict(aggregation='sum'))

def check_site_weights(f, reductions):
    mix = dict(rates=[1, 2], prior=[0.2, 0.8])
    for rate_mixture in None, mix:
        #
        m = copy.deepcopy(model_a)
        if rate_mixture:
            m['rate_mixture'] = rate_mixture
        in_a = dict(model_and_data = m)
        in_a.update(site_sum_a)
        for r in reductions:
            in_a.update(r)
        #
        m = copy.deepcopy(model_b)
        if rate_mixture:
            m['rate_mixture'] = rate_mixture
        in_b = dict(model_and_data = m)
        in_b.update(site_sum_b)
        for r in reductions:
            in_b.update(r)
        #
        a = json.loads(f(json.dumps(in_a)))
        b = json.loads(f(json.dumps(in_b)))
        assert_equal(a, b)

def test_site_weights():
    for f, reductions in (
        (arbplf_ll, []),
        (arbplf_marginal, [node_sum, state_sum]),
        (arbplf_dwell, [edge_sum, state_sum]),
        (arbplf_trans, [edge_sum, trans_sum]),
        (arbplf_em_update, []),
        (arbplf_newton_delta, []),
        (arbplf_newton_update, []),
        (arbplf_deriv, [edge_sum]),
        (arbplf_hess, []),
        (arbplf_inv_hess, [])):
        print(f)
        yield check_site_weights, f, reductions
