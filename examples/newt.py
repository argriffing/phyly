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

from arbplf import (
        arbplf_ll,
        arbplf_newton_point,
        arbplf_newton_delta,
        arbplf_deriv,
        arbplf_coeff_expect)

def summarize(d):
    print('newton delta:')
    print(arbplf_newton_delta(json.dumps(d)))
    print('deriv:')
    print(arbplf_deriv(json.dumps(d)))
    print('ll:')
    print(arbplf_ll(json.dumps(d)))

def main():
    d = json.loads(sys.stdin.read())
    for i in range(4):
        #s = arbplf_newton_point(json.dumps(d))
        s = arbplf_coeff_expect(json.dumps(d))
        df = pd.read_json(StringIO(s), orient='split', precise_float=True)
        r = list(df.value)
        d['model_and_data']['edge_rate_coefficients'] = r
    print(r)

    summarize(d)

    for i in range(6):
        s = arbplf_newton_point(json.dumps(d))
        df = pd.read_json(StringIO(s), orient='split', precise_float=True)
        r = list(df.value)
        d['model_and_data']['edge_rate_coefficients'] = r
    print(r)

    summarize(d)

main()
