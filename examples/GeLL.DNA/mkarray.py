"""
brown.nuc:
human chimpanzee gorilla orangutan gibbon
0     1          2       3         4

"""
from __future__ import print_function, division

from StringIO import StringIO

import json


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

def get_array_string():
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
    #s = json.dumps(probability_array, indent=2)
    #s = json.dumps(probability_array)
    #print(s)
    f = StringIO()
    print('{ "model_and_data" : { "probability_array" :', file=f)
    print('[', file=f)
    for site, site_data in enumerate(probability_array):
        s = str(site_data).replace(' ', '')
        """
        print('  [')
        for node, node_data in enumerate(site_data):
            print('    ', node_data, sep='', end='')
            if node == len(site_data) - 1:
                print()
            else:
                print(',')
        if site == len(probability_array) - 1:
            print('  ]')
        else:
            print('  ],')
        """
        print('    ', s, sep='', end='', file=f)
        if site == len(probability_array) - 1:
            print('', file=f)
        else:
            print(',', file=f)
    print('] },', file=f)
    print('"site_reduction" : {"aggregation" : "sum"} }', file=f)
    return f.getvalue()

def main():
    print(get_array_string())

if __name__ == '__main__':
    main()
