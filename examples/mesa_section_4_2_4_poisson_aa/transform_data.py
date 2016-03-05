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



def canonical(seq_in):
    d = {}
    seq_out = []
    for x in seq_in:
        if x not in d:
            d[x] = len(d)
        seq_out.append(d[x])
    return seq_out


def get_info(fin):
    aas = ''.join(x for x in string.ascii_uppercase if x not in 'BJOUXZ')
    d = {a : i for i, a in enumerate(aas)}

    sequences = []
    lines = fin.readlines()
    header = lines[0]
    for line in lines[1:]:
        name, sequence = line.strip().split()
        sequences.append([d[x] for x in sequence])

    canonical_columns = defaultdict(int)
    for column in zip(*sequences):
        ccol = canonical(column)
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
            lst = [(1/state_count if (j == x) else 0) for j in states]
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

    weights = [count for ccol, count in pairs]

    return expanded, weights

    #print(len(lines), 'total columns')
    #print(len(canonical_columns), 'columns unique up to residue label')


def main():
    arr, weights = get_probability_array(sys.stdin)
    model_and_data = OrderedDict(
            edges = get_hardcoded_edges(),
            edge_rate_coefficients = get_hardcoded_edge_rate_coefficients(),
            rate_matrix = get_hardcoded_rate_matrix(),
            probability_array = arr)
    js = OrderedDict(
            model_and_data = model_and_data,
            site_reduction = OrderedDict(aggregation=weights))
    print(json.dumps(js))


if __name__ == '__main__':
    main()
