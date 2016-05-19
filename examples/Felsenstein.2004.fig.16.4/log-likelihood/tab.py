from __future__ import print_function

from sys import stdin

from tabulate import tabulate
import pandas as pd

df = pd.read_json(stdin, orient='split', precise_float=True)
tab = tabulate(df, headers='keys', tablefmt='pipe')
print(tab)
