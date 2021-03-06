This is the first test in the
[LikelihoodTest.java](https://github.com/beast-dev/beast-mcmc/blob/master/src/test/dr/evomodel/treelikelihood/LikelihoodTest.java) file in BEAST.
It expects the log likelihood `-1992.20564`
for a sequence alignment from several primate taxa,
given a known tree, under Jukes-Cantor evolution.

Command:
```shell
$ arbplf-ll < in.json > out.json
```

Output (out.json):
```json
{"columns": ["value"], "data": [[-1992.2056440317256]]}
```

Input (in.json):
```json
{
  "model_and_data": {
    "edges": [
      [10, 8],
      [8, 6],
      [6, 4],
      [4, 0],
      [4, 3],
      [3, 1],
      [3, 2],
      [6, 5],
      [8, 7],
      [10, 9]], 
    "edge_rate_coefficients": [
      0.030457, 
      0.033087, 
      0.012035, 
      0.024003, 
      0.013231, 
      0.010772, 
      0.010772, 
      0.036038, 
      0.069125, 
      0.099582],
    "root_prior": [0.25, 0.25, 0.25, 0.25],
    "rate_matrix": [
      [0, 1, 1, 1], 
      [1, 0, 1, 1], 
      [1, 1, 0, 1], 
      [1, 1, 1, 0]],
    "rate_divisor": 3,
    "character_definitions" : [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
      [1, 1, 1, 1]],
    "character_data" : [
      [1, 0, 1, 4, 4, 1, 4, 1, 4, 1, 4],
      [3, 2, 2, 4, 4, 3, 4, 2, 4, 3, 4],
      [2, 2, 2, 4, 4, 2, 4, 2, 4, 3, 4],
      [1, 1, 0, 4, 4, 1, 4, 1, 4, 0, 4],
      [2, 3, 3, 4, 4, 3, 4, 3, 4, 3, 4],
      [0, 1, 0, 4, 4, 1, 4, 1, 4, 0, 4],
      [1, 1, 1, 4, 4, 0, 4, 1, 4, 1, 4],
      [2, 2, 3, 4, 4, 2, 4, 2, 4, 3, 4],
      [0, 0, 1, 4, 4, 0, 4, 1, 4, 0, 4],
      [2, 2, 2, 4, 4, 2, 4, 1, 4, 3, 4],
      [0, 0, 0, 4, 4, 0, 4, 0, 4, 1, 4],
      [3, 2, 2, 4, 4, 3, 4, 3, 4, 3, 4],
      [2, 2, 2, 4, 4, 3, 4, 3, 4, 2, 4],
      [2, 3, 3, 4, 4, 0, 4, 2, 4, 2, 4],
      [1, 0, 0, 4, 4, 1, 4, 1, 4, 1, 4],
      [0, 0, 0, 4, 4, 0, 4, 1, 4, 1, 4],
      [4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4],
      [2, 3, 2, 4, 4, 3, 4, 2, 4, 2, 4],
      [1, 1, 1, 4, 4, 1, 4, 0, 4, 0, 4],
      [1, 1, 1, 4, 4, 1, 4, 1, 4, 2, 4],
      [1, 1, 1, 4, 4, 0, 4, 0, 4, 1, 4],
      [2, 2, 2, 4, 4, 3, 4, 3, 4, 1, 4],
      [2, 2, 2, 4, 4, 2, 4, 3, 4, 2, 4],
      [1, 1, 0, 4, 4, 1, 4, 1, 4, 1, 4],
      [2, 3, 2, 4, 4, 2, 4, 4, 4, 2, 4],
      [0, 0, 0, 4, 4, 1, 4, 1, 4, 0, 4],
      [0, 0, 0, 4, 4, 1, 4, 0, 4, 0, 4],
      [0, 0, 0, 4, 4, 0, 4, 2, 4, 0, 4],
      [2, 2, 3, 4, 4, 2, 4, 2, 4, 2, 4],
      [1, 0, 0, 4, 4, 1, 4, 0, 4, 0, 4],
      [1, 0, 0, 4, 4, 1, 4, 1, 4, 0, 4],
      [2, 2, 2, 4, 4, 3, 4, 3, 4, 3, 4],
      [0, 1, 1, 4, 4, 1, 4, 1, 4, 1, 4],
      [1, 0, 0, 4, 4, 0, 4, 1, 4, 0, 4],
      [1, 0, 0, 4, 4, 0, 4, 0, 4, 0, 4],
      [2, 2, 2, 4, 4, 0, 4, 2, 4, 2, 4],
      [1, 1, 1, 4, 4, 1, 4, 0, 4, 1, 4],
      [2, 2, 2, 4, 4, 2, 4, 3, 4, 3, 4],
      [0, 0, 0, 4, 4, 0, 4, 1, 4, 0, 4],
      [3, 3, 3, 4, 4, 3, 4, 3, 4, 3, 4],
      [0, 0, 0, 4, 4, 0, 4, 1, 4, 3, 4],
      [3, 3, 3, 4, 4, 3, 4, 2, 4, 3, 4],
      [1, 1, 1, 4, 4, 1, 4, 1, 4, 1, 4],
      [2, 2, 2, 4, 4, 2, 4, 2, 4, 2, 4],
      [2, 2, 2, 4, 4, 3, 4, 2, 4, 3, 4],
      [2, 2, 2, 4, 4, 2, 4, 1, 4, 2, 4],
      [4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4],
      [3, 2, 3, 4, 4, 3, 4, 3, 4, 2, 4],
      [4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4],
      [1, 1, 1, 4, 4, 0, 4, 1, 4, 0, 4],
      [1, 1, 1, 4, 4, 0, 4, 0, 4, 0, 4],
      [3, 2, 2, 4, 4, 2, 4, 3, 4, 2, 4],
      [2, 1, 2, 4, 4, 2, 4, 2, 4, 2, 4],
      [2, 2, 2, 4, 4, 2, 4, 4, 4, 2, 4],
      [4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4],
      [0, 1, 1, 4, 4, 1, 4, 0, 4, 1, 4],
      [0, 0, 0, 4, 4, 1, 4, 1, 4, 2, 4],
      [3, 3, 3, 4, 4, 3, 4, 3, 4, 2, 4],
      [2, 2, 2, 4, 4, 3, 4, 2, 4, 2, 4],
      [2, 2, 2, 4, 4, 2, 4, 3, 4, 1, 4],
      [4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4],
      [0, 1, 0, 4, 4, 0, 4, 1, 4, 0, 4],
      [0, 1, 0, 4, 4, 0, 4, 0, 4, 0, 4],
      [0, 0, 0, 4, 4, 0, 4, 0, 4, 0, 4],
      [1, 1, 1, 4, 4, 1, 4, 1, 4, 0, 4],
      [3, 3, 3, 4, 4, 3, 4, 2, 4, 2, 4],
      [3, 2, 2, 4, 4, 2, 4, 2, 4, 2, 4],
      [0, 0, 0, 4, 4, 0, 4, 0, 4, 2, 4],
      [3, 3, 3, 4, 4, 2, 4, 2, 4, 3, 4]]},
  "site_reduction" : {
    "aggregation" : [
      1, 1, 8, 1, 1, 1, 4, 1, 1, 1, 12, 1, 2, 1, 1, 6, 2, 1, 3, 2,
      1, 1, 9, 1, 1, 1, 3, 1, 1, 2, 1, 1, 4, 2, 3, 1, 5, 2, 9, 99,
      1, 2, 148, 221, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 2, 1, 1, 4, 4, 1,
      1, 1, 1, 155, 8, 1, 3, 1, 1]}
}
```
