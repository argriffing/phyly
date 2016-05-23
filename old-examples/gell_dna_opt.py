"""
Test log likelihood using the GeLL test.

https://github.com/danielmoney/GeLL
LikelihoodTest.java

brown.nuc:
human chimpanzee gorilla orangutan gibbon
0     1          2       3         4
"""
from __future__ import print_function, division

from StringIO import StringIO
import json

import scipy
import scipy.stats
from scipy.special import gammainc, logit, expit
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import pandas as pd

from arbplf import arbplf_ll

# Precompute a probability array associated with the brown.nuc alignment.

brown_nuc_sequences = (
"""
AAGCTTCACCGGCGCAGTCATTCTCATAATCGCCCACGGACTTACATCCTCATTACTATT
CTGCCTAGCAAACTCAAACTACGAACGCACTCACAGTCGCATCATAATCCTCTCTCAAGG
ACTTCAAACTCTACTCCCACTAATAGCTTTTTGATGACTTCTAGCAAGCCTCGCTAACCT
CGCCTTACCCCCCACTATTAACCTACTGGGAGAACTCTCTGTGCTAGTAACCACGTTCTC
CTGATCAAATATCACTCTCCTACTTACAGGACTCAACATACTAGTCACAGCCCTATACTC
CCTCTACATATTTACCACAACACAATGGGGCTCACTCACCCACCACATTAACAACATAAA
ACCCTCATTCACACGAGAAAACACCCTCATGTTCATACACCTATCCCCCATTCTCCTCCT
ATCCCTCAACCCCGACATCATTACCGGGTTTTCCTCTTGTAAATATAGTTTAACCAAAAC
ATCAGATTGTGAATCTGACAACAGAGGCTTACGACCCCTTATTTACCGAGAAAGCTCACA
AGAACTGCTAACTCATGCCCCATGTCTGACAACATGGCTTTCTCAACTTTTAAAGGATA
ACAGCTATCCATTGGTCTTAGGCCCCAAAAATTTTGGTGCAACTCCAAATAAAAGTAATA
ACCATGCACACTACTATAACCACCCTAACCCTGACTTCCCTAATTCCCCCCATCCTTACC
ACCCTCGTTAACCCTAACAAAAAAAACTCATACCCCCATTATGTAAAATCCATTGTCGCA
TCCACCTTTATTATCAGTCTCTTCCCCACAACAATATTCATGTGCCTAGACCAAGAAGTT
ATTATCTCGAACTGACACTGAGCCACAACCCAAACAACCCAGCTCTCCCTAAGCTT    
""",
"""
AAGCTTCACCGGCGCAATTATCCTCATAATCGCCCACGGACTTACATCCTCATTATTATT
CTGCCTAGCAAACTCAAATTATGAACGCACCCACAGTCGCATCATAATTCTCTCCCAAGG
ACTTCAAACTCTACTCCCACTAATAGCCTTTTGATGACTCCTAGCAAGCCTCGCTAACCT
CGCCCTACCCCCTACCATTAATCTCCTAGGGGAACTCTCCGTGCTAGTAACCTCATTCTC
CTGATCAAATACCACTCTCCTACTCACAGGATTCAACATACTAATCACAGCCCTGTACTC
CCTCTACATGTTTACCACAACACAATGAGGCTCACTCACCCACCACATTAATAACATAAA
GCCCTCATTCACACGAGAAAATACTCTCATATTTTTACACCTATCCCCCATCCTCCTTCT
ATCCCTCAATCCTGATATCATCACTGGATTCACCTCCTGTAAATATAGTTTAACCAAAAC
ATCAGATTGTGAATCTGACAACAGAGGCTCACGACCCCTTATTTACCGAGAAAGCTTATA
AGAACTGCTAATTCATATCCCATGCCTAACAACATGGCTTTCTCAACTTTTAAAGGATA
ACAGCCATCCGTTGGTCTTAGGCCCCAAAAATTTTGGTGCAACTCCAAATAAAAGTAATA
ACCATGTATACTACCATAACCACCTTAACCCTAACTCCCTTAATTCTCCCCATCCTCACC
ACCCTCATTAACCCTAACAAAAAAAACTCATATCCCCATTATGTGAAATCCATTATCGCG
TCCACCTTTATCATTAGCCTTTTCCCCACAACAATATTCATATGCCTAGACCAAGAAGCT
ATTATCTCAAACTGGCACTGAGCAACAACCCAAACAACCCAGCTCTCCCTAAGCTT    
""",
"""
AAGCTTCACCGGCGCAGTTGTTCTTATAATTGCCCACGGACTTACATCATCATTATTATT
CTGCCTAGCAAACTCAAACTACGAACGAACCCACAGCCGCATCATAATTCTCTCTCAAGG
ACTCCAAACCCTACTCCCACTAATAGCCCTTTGATGACTTCTGGCAAGCCTCGCCAACCT
CGCCTTACCCCCCACCATTAACCTACTAGGAGAGCTCTCCGTACTAGTAACCACATTCTC
CTGATCAAACACCACCCTTTTACTTACAGGATCTAACATACTAATTACAGCCCTGTACTC
CCTTTATATATTTACCACAACACAATGAGGCCCACTCACACACCACATCACCAACATAAA
ACCCTCATTTACACGAGAAAACATCCTCATATTCATGCACCTATCCCCCATCCTCCTCCT
ATCCCTCAACCCCGATATTATCACCGGGTTCACCTCCTGTAAATATAGTTTAACCAAAAC
ATCAGATTGTGAATCTGATAACAGAGGCTCACAACCCCTTATTTACCGAGAAAGCTCGTA
AGAGCTGCTAACTCATACCCCGTGCTTAACAACATGGCTTTCTCAACTTTTAAAGGATA
ACAGCTATCCATTGGTCTTAGGACCCAAAAATTTTGGTGCAACTCCAAATAAAAGTAATA
ACTATGTACGCTACCATAACCACCTTAGCCCTAACTTCCTTAATTCCCCCTATCCTTACC
ACCTTCATCAATCCTAACAAAAAAAGCTCATACCCCCATTACGTAAAATCTATCGTCGCA
TCCACCTTTATCATCAGCCTCTTCCCCACAACAATATTTCTATGCCTAGACCAAGAAGCT
ATTATCTCAAGCTGACACTGAGCAACAACCCAAACAATTCAACTCTCCCTAAGCTT    
""",
"""
AAGCTTCACCGGCGCAACCACCCTCATGATTGCCCATGGACTCACATCCTCCCTACTGTT
CTGCCTAGCAAACTCAAACTACGAACGAACCCACAGCCGCATCATAATCCTCTCTCAAGG
CCTTCAAACTCTACTCCCCCTAATAGCCCTCTGATGACTTCTAGCAAGCCTCACTAACCT
TGCCCTACCACCCACCATCAACCTTCTAGGAGAACTCTCCGTACTAATAGCCATATTCTC
TTGATCTAACATCACCATCCTACTAACAGGACTCAACATACTAATCACAACCCTATACTC
TCTCTATATATTCACCACAACACAACGAGGTACACCCACACACCACATCAACAACATAAA
ACCTTCTTTCACACGCGAAAATACCCTCATGCTCATACACCTATCCCCCATCCTCCTCTT
ATCCCTCAACCCCAGCATCATCGCTGGGTTCGCCTACTGTAAATATAGTTTAACCAAAAC
ATTAGATTGTGAATCTAATAATAGGGCCCCACAACCCCTTATTTACCGAGAAAGCTCACA
AGAACTGCTAACTCTCACTCCATGTGTAACAACATGGCTTTCTCAGCTTTTAAAGGATA
ACAGCTATCCCTTGGTCTTAGGATCCAAAAATTTTGGTGCAACTCCAAATAAAAGTAACA
GCCATGTTTACCACCATAACTGCCCTCACCTTAACTTCCCTAATCCCCCCCATTACCGCT
ACCCTCATTAACCCCAACAAAAAAAACCCATACCCCCACTATGTAAAAACGGCCATCGCA
TCCGCCTTTACTATCAGCCTTATCCCAACAACAATATTTATCTGCCTAGGACAAGAAACC
ATCGTCACAAACTGATGCTGAACAACCACCCAGACACTACAACTCTCACTAAGCTT    
""",
"""
AAGCTTTACAGGTGCAACCGTCCTCATAATCGCCCACGGACTAACCTCTTCCCTGCTATT
CTGCCTTGCAAACTCAAACTACGAACGAACTCACAGCCGCATCATAATCCTATCTCGAGG
GCTCCAAGCCTTACTCCCACTGATAGCCTTCTGATGACTCGCAGCAAGCCTCGCTAACCT
CGCCCTACCCCCCACTATTAACCTCCTAGGTGAACTCTTCGTACTAATGGCCTCCTTCTC
CTGGGCAAACACTACTATTACACTCACCGGGCTCAACGTACTAATCACGGCCCTATACTC
CCTTTACATATTTATCATAACACAACGAGGCACACTTACACACCACATTAAAAACATAAA
ACCCTCACTCACACGAGAAAACATATTAATACTTATGCACCTCTTCCCCCTCCTCCTCCT
AACCCTCAACCCTAACATCATTACTGGCTTTACTCCCTGTAAACATAGTTTAATCAAAAC
ATTAGATTGTGAATCTAACAATAGAGGCTCGAAACCTCTTGCTTACCGAGAAAGCCCACA
AGAACTGCTAACTCACTACCCATGTATAACAACATGGCTTTCTCAACTTTTAAAGGATA
ACAGCTATCCATTGGTCTTAGGACCCAAAAATTTTGGTGCAACTCCAAATAAAAGTAATA
GCAATGTACACCACCATAGCCATTCTAACGCTAACCTCCCTAATTCCCCCCATTACAGCC
ACCCTTATTAACCCCAATAAAAAGAACTTATACCCGCACTACGTAAAAATGACCATTGCC
TCTACCTTTATAATCAGCCTATTTCCCACAATAATATTCATGTGCACAGACCAAGAAACC
ATTATTTCAAACTGACACTGAACTGCAACCCAAACGCTAGAACTCTCCCTAAGCTT    
"""
)

def elem(i):
    x = [0]*4
    x[i] = 1
    return x
probability_array = []
sequences = [''.join(s.split()) for s in brown_nuc_sequences]
state_map = dict(zip('TCAG', (0, 1, 2, 3)))
for column in zip(*sequences):
    rows = []
    for nt in column:
        rows.append(elem(state_map[nt]))
    # 3 internal nodes
    for i in range(3):
        rows.append([1, 1, 1, 1])
    probability_array.append(rows)

# rate mixture
gamma_shape = 0.19249
rate_category_count = 4

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
    for lb, ub in pairs:
        l = f(lb)
        u = f(ub)
        r = u - l
        rates.append(r)
    rates = N * np.array(rates)
    return rates.tolist()

rate_mixture = dict(
        prior = [1/rate_category_count] * rate_category_count,
        rates = discretized_gamma(rate_category_count, gamma_shape))

def objective(X):
    # 7 branch length parameters
    # 3 stationary distribution parameters
    # 5 symmetric GTR parameters
    # 1 gamma rate mixture shape parameter
    print('in:', X)
    #edge_rate_coefficients = np.exp(X[:7]).tolist()
    edge_rate_coefficients = X[:7].tolist()
    p = expit(X[7:7+3])
    q = expit(-X[7:7+3])
    root_prior = np.zeros(4)
    root_prior[0] = p[0] * p[1]
    root_prior[1] = p[0] * q[1]
    root_prior[2] = q[0] * p[2]
    root_prior[3] = q[0] * q[2]
    a, b, c, d, e = np.exp(X[7+3:7+3+5])
    rate_matrix = np.array([
        [0, a, b, c],
        [a, 0, d, e],
        [b, d, 0, 1],
        [c, e, 1, 0]])
    rate_matrix = (rate_matrix * root_prior).tolist()
    root_prior = root_prior.tolist()
    gamma_shape = np.exp(X[-1])
    rate_mixture = dict(
            prior = [1/rate_category_count] * rate_category_count,
            rates = discretized_gamma(rate_category_count, gamma_shape))
    model_and_data = dict(
            edges = [[5, 0], [5, 1], [6, 5], [6, 2], [7, 6], [7, 3], [7, 4]],
            edge_rate_coefficients = edge_rate_coefficients,
            root_prior = root_prior,
            rate_matrix = rate_matrix,
            rate_mixture = rate_mixture,
            probability_array = probability_array)
    d = dict(
        model_and_data = model_and_data,
        site_reduction = dict(aggregation='sum'))
    s = arbplf_ll(json.dumps(d))
    df = pd.read_json(StringIO(s), orient='split', precise_float=True)
    log_likelihood = df.values[0, 0]
    y = -log_likelihood
    print('out:', y)
    return y

def main():
    # define the initial guess
    X0 = np.zeros(7+3+5+1)
    #X0[:7] = np.log([
        #0.057987, 0.074612, 0.035490, 0.074352, 0.131394, 0.350156, 0.544601])
    X0[:7] = [0.057987, 0.074612, 0.035490, 0.074352, 0.131394, 0.350156, 0.544601]
    X0[7:7+3] = np.zeros(3) # equivalent to uniform equilibrium
    X0[7+3:7+3+5] = np.zeros(5) # equivalent to symmetric parts of rates being 1
    X0[-1] = np.log(0.19)
    # minimize the objective
    bounds = [(None, None) for x in X0]
    bounds[:7] = [(1e-6, None) for i in range(7)]
    res = fmin_l_bfgs_b(
            objective, X0, approx_grad=True, bounds=bounds,
            factr=10, pgtol=1e-6)
    print(res)

if __name__ == '__main__':
    main()
