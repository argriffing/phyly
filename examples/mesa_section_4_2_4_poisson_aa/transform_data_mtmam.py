"""
parse files like mtCDNApri.aa from Ziheng Yang's MESA

Some stuff is hardcoded...
There are 5 internal nodes for which no data is available.
Node 0 is the root node, and data is available for this node
as well as for leaf nodes nodes 1..6.
The prior distribution at the root node 0 is uniform over the 20 states.

"""
from __future__ import print_function, division, absolute_import

from collections import defaultdict, OrderedDict
import string
import sys
import json

import numpy as np
from numpy.testing import assert_equal


s_mtmam = """\
 32                                                                         
  2   4                                                                     
 11   0 864                                                                 
  0 186   0   0                                                             
  0 246   8  49   0                                                         
  0   0   0 569   0 274                                                     
 78  18  47  79   0   0  22                                                 
  8 232 458  11 305 550  22   0                                             
 75   0  19   0  41   0   0   0   0                                         
 21   6   0   0  27  20   0   0  26 232                                     
  0  50 408   0   0 242 215   0   0   6   4                                 
 76   0  21   0   0  22   0   0   0 378 609  59                             
  0   0   6   5   7   0   0   0   0  57 246   0  11                         
 53   9  33   2   0  51   0   0  53   5  43  18   0  17                     
342   3 446  16 347  30  21 112  20   0  74  65  47  90 202                 
681   0 110   0 114   0   4   0   1 360  34  50 691   8  78 614             
  5  16   6   0  65   0   0   0   0   0  12   0  13   0   7  17   0         
  0   0 156   0 530  54   0   1 1525 16  25  67   0 682   8 107   0  14    
398   0   0  10   0  33  20   5   0 2220 100  0 832   6   0   0 237   0   0\
"""


s_distn = """
0.0692 0.0184 0.0400 0.0186 0.0065 0.0238 0.0236 0.0557 0.0277 0.0905
0.1675 0.0221 0.0561 0.0611 0.0536 0.0725 0.0870 0.0293 0.0340 0.0428
"""

s_aas = 'ARNDCQEGHILKMFPSTWYV'

def get_rate_matrix():
    nstates = len(s_aas)
    assert_equal(nstates, 20)
    d = {a : i for i, a in enumerate(s_aas)}
    distn = [float(x) for x in s_distn.strip().split()]
    assert_equal(len(distn), nstates)
    lines = s_mtmam.splitlines()
    assert_equal(len(lines), nstates-1)
    rate_matrix = np.zeros((nstates, nstates), dtype=int)
    for i, line in enumerate(lines):
        row_index = i + 1
        row = [int(x) for x in line.strip().split()]
        assert_equal(len(row), row_index)
        rate_matrix[row_index, :row_index] = row
    rate_matrix = np.multiply(rate_matrix + rate_matrix.T, distn)
    np.fill_diagonal(rate_matrix, 0)
    return rate_matrix.tolist()


def canonical(seq_in):
    # FIXME: This code assumed a Poisson rate matrix.
    #        With an 'MTMAM' rate matrix this function is obsolete.
    assert False
    d = {}
    seq_out = []
    for x in seq_in:
        if x not in d:
            d[x] = len(d)
        seq_out.append(d[x])
    return seq_out


def get_info(fin):
    # aas = ''.join(x for x in string.ascii_uppercase if x not in 'BJOUXZ')
    aas = s_aas
    d = {a : i for i, a in enumerate(aas)}

    sequences = []
    lines = fin.readlines()
    header = lines[0]
    for line in lines[1:]:
        name, sequence = line.strip().split()
        sequences.append([d[x] for x in sequence])

    canonical_columns = defaultdict(int)
    for column in zip(*sequences):
        # ccol = canonical(column)
        ccol = column
        canonical_columns[tuple(ccol)] += 1

    return sequences, canonical_columns


def get_hardcoded_edges():
    edges = [
            [0, 8],
            [7, 1],
            [7, 2],
            [8, 7],
            [8, 9],
            [9, 3],
            [9, 10],
            [10, 6],
            [10, 11],
            [11, 4],
            [11, 5]]
    return edges


def get_hardcoded_rate_matrix():
    state_count = 20
    arr = []
    for i in range(state_count):
        row = [1] * state_count
        row[i] = 0
        arr.append(row)
    return arr

def get_hardcoded_edge_rate_coefficients():
    state_count = 11
    return [0.001] * state_count

def get_probability_array(fin):
    sequences, canonical_columns = get_info(fin)

    distn = [float(x) for x in s_distn.strip().split()]

    lines = []
    for column in zip(*sequences):
        lines.append(str(list(column)))

    #print('all columns:')
    #print(',\n'.join(lines))
    #print()

    canonical_lines = []
    counts = []
    pairs = sorted(canonical_columns.items())
    for ccol, count in pairs:
        canonical_lines.append(str(list(ccol)))
        counts.append(count)

    #print('canonical columns and corresponding counts:')
    #print(',\n'.join(canonical_lines))
    #print(counts)
    #print(range(len(counts)))
    #print()

    expanded = []
    for ccol, count in pairs:
        arr = []
        # constants
        internal_node_count = 5
        state_count = 20
        # root node
        states = range(state_count)
        for i, x in enumerate(ccol[:1]):
            # lst = [(1/state_count if (j == x) else 0) for j in states]
            lst = [(distn[j] if (j == x) else 0) for j in states]
            arr.append(lst)
        # other leaf nodes
        for i, x in enumerate(ccol[1:]):
            lst = [(1 if (j == x) else 0) for j in states]
            arr.append(lst)
        # internal nodes
        for i in range(internal_node_count):
            lst = [1] * state_count
            arr.append(lst)
        expanded.append(arr)
    #print('expanded canonical columns:')
    #for x in expanded:
        #print(x)
    #print()

    # total_count = sum(count for ccol, count in pairs)

    # weights = [count / total_count for ccol, count in pairs]
    weights = [count for ccol, count in pairs]

    return expanded, weights

    #print(len(lines), 'total columns')
    #print(len(canonical_columns), 'columns unique up to residue label')


def main():
    arr, weights = get_probability_array(sys.stdin)
    model_and_data = OrderedDict(
            edges = get_hardcoded_edges(),
            edge_rate_coefficients = get_hardcoded_edge_rate_coefficients(),
            # rate_matrix = get_hardcoded_rate_matrix(),
            rate_matrix = get_rate_matrix(),
            probability_array = arr)
    js = OrderedDict(
            model_and_data = model_and_data,
            site_reduction = OrderedDict(aggregation=weights))
    print(json.dumps(js))


if __name__ == '__main__':
    main()


