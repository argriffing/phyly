"""
See empiricalStateFrequenciesPAUP in
https://github.com/beast-dev/beast-mcmc/blob/master/src/dr/evolution/alignment/PatternList.java

"""
from __future__ import print_function, division

import numpy as np

def get_prior(observations, state_count):
    """
    None is treated as a 'missing' observation.

    """
    states = range(state_count)
    prior = np.ones(state_count) / state_count
    for i in range(1000):
        counts = np.zeros(state_count)
        for x in observations:
            if x in states:
                counts[x] += 1
            else:
                counts += prior
        post = counts / counts.sum()
        err = np.absolute(prior - post).sum()
        if err < 1e-8:
            break
        prior = post
    return prior.tolist()
