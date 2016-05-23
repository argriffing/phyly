from __future__ import print_function, division

from StringIO import StringIO
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arbplf import arbplf_dwell

def mydwell(d):
    s = arbplf_dwell(json.dumps(d))
    df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    return df

def main():
    ts = np.linspace(1e-5, 30, 100)
    n = len(ts)
    arr = []
    for i, rate in enumerate(ts):
        model_and_data = {
                "edges" : [[0, 1]],
                "edge_rate_coefficients" : [0.5 * rate],
                "rate_matrix" : [
                    [0, 1, 0, 0],
                    [0, 0, 2, 0],
                    [0, 0, 0, 3],
                    [0, 0, 0, 0]],
                "probability_array" : [[
                    [1, 0, 0, 0],
                    [1, 1, 1, 1]]]}
        d = {
                "model_and_data" : model_and_data,
                "site_reduction" : {"aggregation" : "sum"},
                "edge_reduction" : {"aggregation" : "sum"}}
        arr.append(mydwell(d)['value'].values.tolist())
    a, b, c, d = np.array(arr).T.tolist()
    lines = plt.plot(
            ts, a, 'blue',
            ts, b, 'green',
            ts, c, 'red',
            ts, d, 'skyblue')
    plt.ylabel("Time-averaged Expected sojourn time")
    plt.xlabel("Time")

    # Use a transparent legend frame.
    plt.legend(
            lines,
            ('State 1', 'State 2', 'State 3', 'State 4 (absorbing)'),
            loc='center right',
            framealpha=0)

    # Use a transparent background for the figure.
    plt.savefig('out00.svg', transparent=True)

main()
