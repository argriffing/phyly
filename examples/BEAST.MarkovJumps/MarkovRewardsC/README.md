This example is taken from `testMarkovJumpsReward` in
[MarkovJumpsSubstitutionModelTest.java](https://github.com/beast-dev/beast-mcmc/blob/master/src/test/dr/evomodel/substmodel/MarkovJumpsSubstitutionModelTest.java)
which computes the an accumulated 'reward' according
to how much time (or what proportion of time?) is spent in which state.

The CTMC process is an HKY model with kappa = 2 and with
the following stationary distribution.
```
p(A) = 0.3
p(C) = 0.2
p(G) = 0.25
p(T) = 0.25
```

This test uses an elapsed time of 1 and a reward of 1 for each state,
so it is less informative than it might otherwise be.
Regardless of the states at the endpoints of a branch,
the conditionally expected 'reward' along the branch is always 1.
The BEAST test has a vector of ones named `rMarkovRewardsC`
which is used as a reference result.

Analysis using arbplf-dwell
---

Command:
```shell
$ arbplf-dwell < in.json > out.json
```

Output (out.json):
```json
{
  "columns": ["site", "value"],
  "data": [
    [0, 1.0],
    [1, 1.0],
    [2, 1.0],
    [3, 1.0],
    [4, 1.0],
    [5, 1.0],
    [6, 1.0],
    [7, 1.0],
    [8, 1.0],
    [9, 1.0],
    [10, 1.0],
    [11, 1.0],
    [12, 1.0],
    [13, 1.0],
    [14, 1.0],
    [15, 1.0]]
}
```

Input (in.json):
```json
{
  "model_and_data": {
    "edges": [[0, 1]],
    "edge_rate_coefficients": [1],
    "root_prior" : [0.3, 0.2, 0.25, 0.25],
    "rate_matrix": [
      [0, 0.2, 0.5, 0.25],
      [0.3, 0, 0.25, 0.5],
      [0.6, 0.2, 0, 0.25],
      [0.3, 0.4, 0.25, 0]],
    "rate_divisor": "equilibrium_exit_rate",
    "character_definitions" : [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
      [1, 1, 1, 1]],
    "character_data" : [
      [0, 0], [0, 1], [0, 2], [0, 3], 
      [1, 0], [1, 1], [1, 2], [1, 3], 
      [2, 0], [2, 1], [2, 2], [2, 3], 
      [3, 0], [3, 1], [3, 2], [3, 3]]},
    "state_reduction" : {"aggregation" : "sum"},
    "edge_reduction" : {"aggregation" : "only"}
}
```
