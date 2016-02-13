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
from arbplf import arbplf_newton_point


def myll(d):
    s = arbplf_ll(json.dumps(d))
    df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    return df

def myderiv(d):
    s = arbplf_deriv(json.dumps(d))
    df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    return df

def main():
    d_in = json.loads(sys.stdin.read())
    print(d_in)
    
    d = copy.deepcopy(d_in)
    d['site_reduction'] = {'aggregation' : 'sum'}
    x = myll(d).value[0]
    print(x)

    d = copy.deepcopy(d_in)
    d['site_reduction'] = {'aggregation' : 'sum'}
    x = myderiv(d).set_index('edge').value
    print(x.values)

    def my_objective(X):
        print('X for objective:', X)
        d = copy.deepcopy(d_in)
        d['model_and_data']['edge_rate_coefficients'] = X.tolist()
        d['site_reduction'] = {'aggregation' : 'sum'}
        y = myll(d).value[0]
        print('y from objective:', y)
        return -y

    def my_gradient(X):
        print('X for gradient:', X)
        d = copy.deepcopy(d_in)
        d['model_and_data']['edge_rate_coefficients'] = X.tolist()
        d['site_reduction'] = {'aggregation' : 'sum'}
        y = myderiv(d).set_index('edge').value.values
        print('y from gradient:', y)
        return -y


    x0 = d_in['model_and_data']['edge_rate_coefficients']
    print(x0)
    
    bounds = [(0, None)]*len(x0)
    result = minimize(my_objective, x0, jac=my_gradient,
            method='l-bfgs-b', bounds=bounds)
    print(result)


main()
