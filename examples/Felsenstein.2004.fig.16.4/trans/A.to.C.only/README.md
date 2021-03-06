Compute the conditionally expected number of `A -> C` substitutions
on each branch, using the example in Figure 16.4 of
Inferring Phylogenies (2004) by Joseph Felsenstein.

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
	 [[1, 1, 1, 1],
	  [1, 1, 1, 1],
	  [1, 1, 1, 1],
	  [1, 1, 1, 1],
	  [1, 1, 1, 1],
	  [1, 1, 1, 1],
	  [1, 1, 1, 1],
	  [1, 1, 1, 1]],
	 [[1, 0, 0, 0],
	  [0, 1, 0, 0],
	  [0, 1, 0, 0],
	  [0, 1, 0, 0],
	  [0, 0, 1, 0],
	  [1, 1, 1, 1],
	  [1, 1, 1, 1],
	  [1, 1, 1, 1]],
	 [[1, 0, 0, 0],
	  [0, 1, 0, 0],
	  [0, 1, 0, 0],
	  [0, 1, 0, 0],
	  [0, 0, 1, 0],
	  [0, 0, 1, 0],
	  [0, 0, 1, 0],
	  [1, 0, 0, 0]]]
  },
   "trans_reduction" : {"selection" : [[0, 1]], "aggregation" : "only"}
}
```

Commands:
```bash
$ arbplf-trans < in.json > out.json
$ tabby < out.json > out.md
```

Output (out.json):
```json
{
  "columns": ["site", "edge", "value"],
  "data": [
    [0, 0, 0.00083333333333333339],
    [0, 1, 0.016666666666666666],
    [0, 2, 0.012499999999999999],
    [0, 3, 0.024999999999999998],
    [0, 4, 0.0041666666666666666],
    [0, 5, 0.024999999999999998],
    [0, 6, 0.0016666666666666668],
    [1, 0, 4.6121539506086039e-6],
    [1, 1, 0.69738556106785288],
    [1, 2, 0.23771960832022732],
    [1, 3, 0.1653848731397928],
    [1, 4, 0.0075213904632242101],
    [1, 5, 0.070267332594395424],
    [1, 6, 0.00014833122092301447],
    [2, 0, 5.5678597803732881e-6],
    [2, 1, 0.93703265488117238],
    [2, 2, 0.02458361084682513],
    [2, 3, 0.048337760914013178],
    [2, 4, 7.8432947543917953e-7],
    [2, 5, 0.048337760914013178],
    [2, 6, 4.971058893110001e-8]]
}
```

Output rendered as a table (out.md):

|    |   site |   edge |                  value |
|---:|-------:|-------:|-----------------------:|
|  0 |      0 |      0 | 0.00083333333333333339 |
|  1 |      0 |      1 | 0.016666666666666666   |
|  2 |      0 |      2 | 0.012499999999999999   |
|  3 |      0 |      3 | 0.024999999999999998   |
|  4 |      0 |      4 | 0.0041666666666666666  |
|  5 |      0 |      5 | 0.024999999999999998   |
|  6 |      0 |      6 | 0.0016666666666666668  |
|  7 |      1 |      0 | 4.6121539506086039e-06 |
|  8 |      1 |      1 | 0.69738556106785288    |
|  9 |      1 |      2 | 0.23771960832022732    |
| 10 |      1 |      3 | 0.1653848731397928     |
| 11 |      1 |      4 | 0.0075213904632242101  |
| 12 |      1 |      5 | 0.070267332594395424   |
| 13 |      1 |      6 | 0.00014833122092301447 |
| 14 |      2 |      0 | 5.5678597803732881e-06 |
| 15 |      2 |      1 | 0.93703265488117238    |
| 16 |      2 |      2 | 0.02458361084682513    |
| 17 |      2 |      3 | 0.048337760914013178   |
| 18 |      2 |      4 | 7.8432947543917953e-07 |
| 19 |      2 |      5 | 0.048337760914013178   |
| 20 |      2 |      6 | 4.971058893110001e-08  |
