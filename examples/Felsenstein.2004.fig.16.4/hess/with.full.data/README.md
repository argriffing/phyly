Compute the second derivatives of log likelihood with respect to
the edge rate scaling factors, following the example in Figure 16.4 of
Inferring Phylogenies (2004) by Joseph Felsenstein.

This analysis assumes that the nucleotide state is known at all
taxa (including leaves and internal nodes) without ambiguity.
Therefore the likelihood is a multiplicatively separable
function of the edge rate coefficient parameters,
so the cross-derivatives of the log likelihood are zero.

---

Input (in.json):
```json
{
"model_and_data" :
  {
  "edges" : [[7, 0], [7, 1], [7, 6], [6, 2], [6, 5], [5, 3], [5, 4]],
  "edge_rate_coefficients" : [0.01, 0.2, 0.15, 0.3, 0.05, 0.3, 0.02],
  "root_prior" : [0.25, 0.25, 0.25, 0.25],
  "rate_matrix" : [
	 [0, 1, 1, 1],
	 [1, 0, 1, 1],
	 [1, 1, 0, 1],
	 [1, 1, 1, 0]],
  "rate_divisor" : 3,
  "probability_array" : [
	 [[1, 0, 0, 0],
	  [0, 1, 0, 0],
	  [0, 1, 0, 0],
	  [0, 1, 0, 0],
	  [0, 0, 1, 0],
	  [0, 0, 1, 0],
	  [0, 0, 1, 0],
	  [1, 0, 0, 0]]]
   },
"site_reduction" : {"aggregation" : "only"}
}
```

Commands:
```bash
$ arbplf-hess < in.json > out.json
$ tabby < out.json > out.md
```

Output (out.json):
```json
{
  "columns": ["first_edge", "second_edge", "value"],
  "data": [
    [0, 0, 0.33555176937041531],
    [0, 1, 0.0],
    [0, 2, 0.0],
    [0, 3, 0.0],
    [0, 4, 0.0],
    [0, 5, 0.0],
    [0, 6, 0.0],
    [1, 0, 0.0],
    [1, 1, -24.852377118097195],
    [1, 2, 0.0],
    [1, 3, 0.0],
    [1, 4, 0.0],
    [1, 5, 0.0],
    [1, 6, 0.0],
    [2, 0, 0.0],
    [2, 1, 0.0],
    [2, 2, -44.296592122938598],
    [2, 3, 0.0],
    [2, 4, 0.0],
    [2, 5, 0.0],
    [2, 6, 0.0],
    [3, 0, 0.0],
    [3, 1, 0.0],
    [3, 2, 0.0],
    [3, 3, -10.964140665084933],
    [3, 4, 0.0],
    [3, 5, 0.0],
    [3, 6, 0.0],
    [4, 0, 0.0],
    [4, 1, 0.0],
    [4, 2, 0.0],
    [4, 3, 0.0],
    [4, 4, 0.34434145686318041],
    [4, 5, 0.0],
    [4, 6, 0.0],
    [5, 0, 0.0],
    [5, 1, 0.0],
    [5, 2, 0.0],
    [5, 3, 0.0],
    [5, 4, 0.0],
    [5, 5, -10.964140665084933],
    [5, 6, 0.0],
    [6, 0, 0.0],
    [6, 1, 0.0],
    [6, 2, 0.0],
    [6, 3, 0.0],
    [6, 4, 0.0],
    [6, 5, 0.0],
    [6, 6, 0.33776230171912619]]
}
```

Output rendered as a table (out.md):

|    |   first_edge |   second_edge |                 value |
|---:|-------------:|--------------:|----------------------:|
|  0 |            0 |             0 |   0.33555176937041531 |
|  1 |            0 |             1 |   0                   |
|  2 |            0 |             2 |   0                   |
|  3 |            0 |             3 |   0                   |
|  4 |            0 |             4 |   0                   |
|  5 |            0 |             5 |   0                   |
|  6 |            0 |             6 |   0                   |
|  7 |            1 |             0 |   0                   |
|  8 |            1 |             1 | -24.852377118097195   |
|  9 |            1 |             2 |   0                   |
| 10 |            1 |             3 |   0                   |
| 11 |            1 |             4 |   0                   |
| 12 |            1 |             5 |   0                   |
| 13 |            1 |             6 |   0                   |
| 14 |            2 |             0 |   0                   |
| 15 |            2 |             1 |   0                   |
| 16 |            2 |             2 | -44.296592122938598   |
| 17 |            2 |             3 |   0                   |
| 18 |            2 |             4 |   0                   |
| 19 |            2 |             5 |   0                   |
| 20 |            2 |             6 |   0                   |
| 21 |            3 |             0 |   0                   |
| 22 |            3 |             1 |   0                   |
| 23 |            3 |             2 |   0                   |
| 24 |            3 |             3 | -10.964140665084933   |
| 25 |            3 |             4 |   0                   |
| 26 |            3 |             5 |   0                   |
| 27 |            3 |             6 |   0                   |
| 28 |            4 |             0 |   0                   |
| 29 |            4 |             1 |   0                   |
| 30 |            4 |             2 |   0                   |
| 31 |            4 |             3 |   0                   |
| 32 |            4 |             4 |   0.34434145686318041 |
| 33 |            4 |             5 |   0                   |
| 34 |            4 |             6 |   0                   |
| 35 |            5 |             0 |   0                   |
| 36 |            5 |             1 |   0                   |
| 37 |            5 |             2 |   0                   |
| 38 |            5 |             3 |   0                   |
| 39 |            5 |             4 |   0                   |
| 40 |            5 |             5 | -10.964140665084933   |
| 41 |            5 |             6 |   0                   |
| 42 |            6 |             0 |   0                   |
| 43 |            6 |             1 |   0                   |
| 44 |            6 |             2 |   0                   |
| 45 |            6 |             3 |   0                   |
| 46 |            6 |             4 |   0                   |
| 47 |            6 |             5 |   0                   |
| 48 |            6 |             6 |   0.33776230171912619 |
