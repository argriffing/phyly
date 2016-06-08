Input (in.json):
```json
{
"model_and_data" :
  {
  "edges" : [[4, 0], [4, 1], [5, 4], [5, 2], [5, 3]],
  "edge_rate_coefficients" : [0.01, 0.02, 0.03, 0.01, 0.10],
  "root_prior" : [0.25, 0.25, 0.25, 0.25],
  "rate_matrix" : [
	 [0, 1, 3, 1],
	 [1, 0, 1, 3],
	 [3, 1, 0, 1],
	 [1, 3, 1, 0]],
  "rate_divisor" : 5,
  "gamma_rate_mixture" : {"gamma_shape" : 1, "gamma_categories" : 4},
  "probability_array" : [
	 [[1, 0, 0, 0],
	  [0, 0, 1, 0],
	  [0, 1, 0, 0],
	  [1, 0, 0, 0],
	  [1, 1, 1, 1],
	  [1, 1, 1, 1]],
	 [[1, 0, 0, 0],
	  [1, 0, 0, 0],
	  [0, 0, 0, 1],
	  [1, 0, 0, 0],
	  [1, 1, 1, 1],
	  [1, 1, 1, 1]],

      ...

	 [[0, 1, 0, 0],
	  [0, 1, 0, 0],
	  [0, 0, 1, 0],
	  [1, 0, 0, 0],
	  [1, 1, 1, 1],
	  [1, 1, 1, 1]]]
   },
   "site_reduction" : {"aggregation" : "sum"}
}
```

Output (out.json):
```json
{
  "columns": ["value"],
  "data": [
    [-85.030942032051257]]
}
```

Negative log likelihoods computed by bpp-phyl:

using single tree traversal:
```
85.030942031997312824
```

using double tree traversal:
```
85.03094203199732703524
```
