This example is taken from `testMarginalRates` in
[MarkovJumpsSubstitutionModelTest.java](https://github.com/beast-dev/beast-mcmc/blob/master/src/test/dr/evomodel/substmodel/MarkovJumpsSubstitutionModelTest.java)
which compares a 'marginal rate' to the reference value
`rMarkovMarginalRate = 0.2010050 * 0.3`.

The 'marginal rate' in question is the
number of `A -> C` substitutions expected at stationary
along a branch where the total expected number of substitutions is 1,
for an HKY model with kappa = 2 and with stationary distribution
```
p(A) = 0.3
p(C) = 0.2
p(G) = 0.25
p(T) = 0.25
```

This expectation is `12 / 199` which is near the BEAST reference value.

```python
>>> 0.201005 * 0.3
0.060301499999999994
>>> 12 / 199
0.06030150753768844
```


Solution using arbplf-trans
---

Command:
```shell
$ arblpf-trans < in.json > out.json
```

Output (out.json):
```json
{
  "columns": ["site", "edge", "first_state", "second_state", "value"],
  "data": [[0, 0, 0, 1, 0.060301507537688447]]
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
    "probability_array" : [[
      [1, 1, 1, 1],
      [1, 1, 1, 1]]]},
    "trans_reduction" : {"selection" : [[0, 1]]}
}
```

A super pedantic way to do it
---

In this variation, the input json file has been
rewritten in a way that is mathematically equivalent but which avoids
numbers not representable exactly in double precision floating point
(for example 0.3 is not exactly represented, but 2.5 is).
This means that the output should be correct within one 'ulp'.

Command:
```shell
$ arblpf-trans < in.pedantic.json > out.pedantic.json
```

Output (out.pedantic.json):
```json
{
  "columns": ["site", "edge", "first_state", "second_state", "value"],
  "data": [[0, 0, 0, 1, 0.06030150753768844]]
}
```

Input (in.pedantic.json):
```json
{
  "model_and_data": {
    "edges": [[0, 1]],
    "edge_rate_coefficients": [1],
    "root_prior" : "equilibrium_distribution",
    "rate_matrix": [
      [0, 2, 5, 2.5],
      [3, 0, 2.5, 5],
      [6, 2, 0, 2.5],
      [3, 4, 2.5, 0]],
    "rate_divisor": "equilibrium_exit_rate",
    "probability_array" : [[
      [1, 1, 1, 1],
      [1, 1, 1, 1]]]},
    "trans_reduction" : {"selection" : [[0, 1]]}
}
```
