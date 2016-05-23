import sys
import pandas as pd
with open(sys.argv[1]) as fin:
    x = pd.read_json(fin, orient='split')
print x
