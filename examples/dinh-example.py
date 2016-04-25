"""
Reference:
"The shape of the one-dimensional phylogenetic likelihood function"
Vu Dinh and Frederick Matsen.
http://arxiv.org/pdf/1507.03647v1.pdf
Figure 1.

"""
from __future__ import print_function, division

from StringIO import StringIO
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arbplf import arbplf_ll, arbplf_newton_refine

def get_json_input(t):
    model_and_data = {
     "edges" : [[0, 1]],
     "edge_rate_coefficients" : [t],
     "rate_matrix" : [
         [0, 3, 1, 1],
         [3, 0, 1, 1],
         [1, 1, 0, 3],
         [1, 1, 3, 0]],
     "rate_divisor" : 8,
     "probability_array" : [
         [[0.24977275, 0.34067358, 0.2051904, 0.20436327], [1, 0, 0, 0]],
         [[0.25, 0.16087344, 0.29328435, 0.29584221], [1, 0, 0, 0]]
     ]}
    d = {
            "model_and_data" : model_and_data,
            "site_reduction" : {"aggregation" : "sum"}}
    return json.dumps(d)

def main():
    xs = np.linspace(1e-5, 1, 100)
    ts = -2 * np.log(xs)
    arr = []
    for i, t in enumerate(ts):
        s = arbplf_ll(get_json_input(t))
        df = pd.read_json(StringIO(s), orient='split', precise_float=True)
        arr.append(df.value.values[0])
    lines = plt.plot(xs, arr, 'blue')
    plt.ylabel("log likelihood")
    plt.xlabel("x = exp(-0.5 t)")
    plt.savefig('out00.svg', transparent=True)

    # local optima
    for i, t in enumerate((0.1, 6.0)):
        s = arbplf_newton_refine(get_json_input(t))
        df = pd.read_json(StringIO(s), orient='split', precise_float=True)
        u = df.value.values[0]
        print('local optimum', i, ':')
        print('  initial guess:', t)
        print('  refined isolated interior local optimum:')
        print('    t = {:.16}'.format(u))
        print('    x =', np.exp(-0.5 * u))

main()
