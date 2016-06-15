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
    sequences = [''.join(s.split()) for s in brown_nuc_sequences]
    state_map = dict(zip('TCAG', (0, 1, 2, 3)))
    character_data = []
    for column in zip(*sequences):
        x = [state_map[nt] for nt in column]
        # 3 internal nodes each with an uninformative character
        x.extend([4]*3)
        character_data.append(x)
    f = StringIO()
    print('{ "model_and_data" : {', file=f)
    print('"character_definitions" : [', file=f)
    print("""
	  [1, 0, 0, 0],
	  [0, 1, 0, 0],
	  [0, 0, 1, 0],
	  [0, 0, 0, 1],
	  [1, 1, 1, 1]],""", file=f)
    print('"character_data" : [', file=f)
    for site, site_data in enumerate(character_data):
        print(site_data, end='', file=f)
        if site == len(character_data) - 1:
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
