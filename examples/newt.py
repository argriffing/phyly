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
        arbplf_newton_point,
        arbplf_newton_delta,
        arbplf_deriv,
        arbplf_coeff_expect)

def main():
    d = json.loads(sys.stdin.read())
    for i in range(1000):
        #s = arbplf_newton_point(json.dumps(d))
        s = arbplf_coeff_expect(json.dumps(d))
        df = pd.read_json(StringIO(s), orient='split', precise_float=True)
        r = list(df.value)
        print(r)
        d['model_and_data']['edge_rate_coefficients'] = r

    print('newton delta:')
    print(arbplf_newton_delta(json.dumps(d)))
    #df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    #r = list(df.value)
    #print(r)

    print('deriv:')
    print(arbplf_deriv(json.dumps(d)))
    #s = arbplf_deriv(json.dumps(d))
    #df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    #r = list(df.value)
    #print(r)

main()
