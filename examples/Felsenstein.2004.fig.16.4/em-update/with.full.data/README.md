Compute one EM update of the edge rate coefficients,
following the setup of Figure 16.4 of
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
$ arbplf-em-update < in.json > out.json
$ tabby < out.json > out.md
```

Output (out.json):
```json
{
  "columns": ["edge", "value"],
  "data": [
    [0, 3.3444318926498214e-5],
    [1, 1.0725855811432092],
    [2, 1.0533311132253989],
    [3, 1.1132979126878946],
    [4, 0.0008471385778404994],
    [5, 1.1132979126878946],
    [6, 0.00013422018084473594]]
}
```

Output rendered as a table (out.md):

|    |   edge |                  value |
|---:|-------:|-----------------------:|
|  0 |      0 | 3.3444318926498214e-05 |
|  1 |      1 | 1.0725855811432092     |
|  2 |      2 | 1.0533311132253989     |
|  3 |      3 | 1.1132979126878946     |
|  4 |      4 | 0.0008471385778404994  |
|  5 |      5 | 1.1132979126878946     |
|  6 |      6 | 0.00013422018084473594 |
