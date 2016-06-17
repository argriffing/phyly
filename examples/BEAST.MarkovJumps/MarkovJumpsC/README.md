This example is taken from `testMarkovJumpsCounts` in
[MarkovJumpsSubstitutionModelTest.java](https://github.com/beast-dev/beast-mcmc/blob/master/src/test/dr/evomodel/substmodel/MarkovJumpsSubstitutionModelTest.java)
which computes the expected number of `A -> C` substitutions
along a branch of length 1, conditional on the endpoint states.
The CTMC process is an HKY model with kappa = 2 and with
the following stationary distribution.
```
p(A) = 0.3
p(C) = 0.2
p(G) = 0.25
p(T) = 0.25
```

Analysis using arbplf-trans
---

Command:
```shell
$ arbplf-trans < in.json > out.json
```

Output (out.json):
```json
{
  "columns": ["site", "edge", "first_state", "second_state", "value"],
  "data": [
    [0, 0, 0, 1, 0.034557322625747726],
    [1, 0, 0, 1, 0.66635306919144355],
    [2, 0, 0, 1, 0.061024826428408126],
    [3, 0, 0, 1, 0.141775071656737],
    [4, 0, 0, 1, 0.0099347016603706789],
    [5, 0, 0, 1, 0.038501747671765015],
    [6, 0, 0, 1, 0.0099347016603706789],
    [7, 0, 0, 1, 0.011499342473472427],
    [8, 0, 0, 1, 0.011561873239441333],
    [9, 0, 0, 1, 0.20159458430087887],
    [10, 0, 0, 1, 0.0060246903175809443],
    [11, 0, 0, 1, 0.028236718722494548],
    [12, 0, 0, 1, 0.0099347016603706789],
    [13, 0, 0, 1, 0.086710476457712418],
    [14, 0, 0, 1, 0.0099347016603706789],
    [15, 0, 0, 1, 0.0057448043879856107]]
}
```

Input (in.json):
```json
{
  "model_and_data": {
    "edges": [[0, 1]],
    "edge_rate_coefficients": [1],
    "root_prior" : [0.2, 0.3, 0.25, 0.25],
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
      [0, 0, 0, 1]],
    "character_data" : [
      [0, 0], [0, 1], [0, 2], [0, 3],
      [1, 0], [1, 1], [1, 2], [1, 3],
      [2, 0], [2, 1], [2, 2], [2, 3],
      [3, 0], [3, 1], [3, 2], [3, 3]]},
  "trans_reduction" : {
    "selection" : [[0, 1]]
  }
}
```

Reordered analysis, with improved accuracy and simpler output
---

To match the BEAST reference values,
we reorder the nucleotides from ACGT to AGCT.
This is analogous to `MarkovJumpsCore.makeComparableToRPackage(c);` in the BEAST test.

A few other minor changes are made:
 - The rate matrix is multiplied by 10 so that its entries
are exactly represented by floating point numbers.
 - A few `"aggregation" : "only"` reductions are added,
to reduce the spamminess of the output JSON file.

Command:
```shell
$ arbplf-trans < in.reordered.json > out.reordered.json
```

Output (out.reordered.json):
```json
{
  "columns": ["site", "value"],
  "data": [
    [0, 0.034557322625747726],
    [1, 0.061024826428408126],
    [2, 0.66635306919144355],
    [3, 0.141775071656737],
    [4, 0.011561873239441333],
    [5, 0.0060246903175809443],
    [6, 0.20159458430087887],
    [7, 0.028236718722494548],
    [8, 0.0099347016603706789],
    [9, 0.0099347016603706789],
    [10, 0.038501747671765015],
    [11, 0.011499342473472427],
    [12, 0.0099347016603706789],
    [13, 0.0099347016603706789],
    [14, 0.086710476457712432],
    [15, 0.0057448043879856107]]
}
```

BEAST reference values (
[rMarkovJumpsC](https://github.com/beast-dev/beast-mcmc/blob/master/src/test/dr/evomodel/substmodel/MarkovJumpsSubstitutionModelTest.java#L141)):
```
0.034557323
0.061024826
0.66635307
0.141775072
0.011561873
0.006024690
0.20159458
0.028236719
0.009934702
0.009934702
0.03850175
0.011499342
0.009934702
0.009934702
0.08671048
0.005744804
```

Input (in.reordered.json):
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
    "character_definitions" : [
      [1, 0, 0, 0],
      [0, 0, 1, 0],
      [0, 1, 0, 0],
      [0, 0, 0, 1]],
    "character_data" : [
      [0, 0], [0, 1], [0, 2], [0, 3],
      [1, 0], [1, 1], [1, 2], [1, 3],
      [2, 0], [2, 1], [2, 2], [2, 3],
      [3, 0], [3, 1], [3, 2], [3, 3]]},
  "edge_reduction" : {"aggregation" : "only"},
  "trans_reduction" : {"selection" : [[0, 1]], "aggregation" : "only"}
}
```
