Example from Figure 16.4 in Inferring Phylogenies (2004)
by Joseph Felsenstein.

```json
{"model_and_data" : {
  "edges" : [[5, 0], [5, 1], [5, 6], [6, 2], [6, 7], [7, 3], [7, 4]],
  "edge_rate_coefficients" : [1, 20, 15, 30, 5, 30, 2],
  "rate_divisor" : 300,
  "root_prior" : [0.25, 0.25, 0.25, 0.25],
  "rate_matrix" : [
	 [0, 1, 1, 1],
	 [1, 0, 1, 1],
	 [1, 1, 0, 1],
	 [1, 1, 1, 0]],
  "probability_array" : [
	 [[1, 0, 0, 0],
	  [0, 1, 0, 0],
	  [0, 1, 0, 0],
	  [0, 1, 0, 0],
	  [0, 0, 1, 0],
	  [1, 1, 1, 1],
	  [1, 1, 1, 1],
	  [1, 1, 1, 1]]]}}
```

```json
{"columns": ["site", "value"], "data": [[0, -11.297288182875496]]}
```
