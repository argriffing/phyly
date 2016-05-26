Compute one EM update of the edge rate coefficients,
following the setup of Figure 16.4 of
Inferring Phylogenies (2004) by Joseph Felsenstein,
but in the complete absence of data.

Because this analysis does not use the available data,
the EM update does not change the edge rate coefficients.

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
    [0, 0.01],
    [1, 0.20000000000000001],
    [2, 0.14999999999999999],
    [3, 0.29999999999999999],
    [4, 0.050000000000000003],
    [5, 0.29999999999999999],
    [6, 0.02]]
}
```

Output rendered as a table (out.md):

|    |   edge |                value |
|---:|-------:|---------------------:|
|  0 |      0 | 0.01                 |
|  1 |      1 | 0.20000000000000001  |
|  2 |      2 | 0.14999999999999999  |
|  3 |      3 | 0.29999999999999999  |
|  4 |      4 | 0.050000000000000003 |
|  5 |      5 | 0.29999999999999999  |
|  6 |      6 | 0.02                 |
