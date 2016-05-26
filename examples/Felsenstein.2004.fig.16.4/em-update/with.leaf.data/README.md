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
	  [1, 1, 1, 1],
	  [1, 1, 1, 1],
	  [1, 1, 1, 1]]]
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
    [0, 0.25807172467207934],
    [1, 0.8265769355851037],
    [2, 0.63897309337333497],
    [3, 0.61878953148237348],
    [4, 0.22784892278613034],
    [5, 0.71897476096295476],
    [6, 0.41552607625292792]]
}
```

Output rendered as a table (out.md):

|    |   edge |               value |
|---:|-------:|--------------------:|
|  0 |      0 | 0.25807172467207934 |
|  1 |      1 | 0.8265769355851037  |
|  2 |      2 | 0.63897309337333497 |
|  3 |      3 | 0.61878953148237348 |
|  4 |      4 | 0.22784892278613034 |
|  5 |      5 | 0.71897476096295476 |
|  6 |      6 | 0.41552607625292792 |
