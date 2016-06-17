```python
>>> log( 0.25 * (0.25 - 0.25 * exp(-2*4/3)) * (0.25 + 0.75 * exp(-4*1/3))**3 )
-5.2555142646496797
```

Command:
```shell
$ arbplf-ll < in.json > out.json
```

Output (out.json):
```json
{"columns": ["site", "value"], "data": [[0, -5.2555142646496797]]}
```

Input (in.json):
```json
{
  "model_and_data": {
    "edges": [[4, 0], [4, 3], [3, 1], [3, 2]],
    "edge_rate_coefficients": [2, 1, 1, 1],
    "root_prior": [0.25, 0.25, 0.25, 0.25],
    "rate_matrix": [
      [0, 1, 1, 1], 
      [1, 0, 1, 1], 
      [1, 1, 0, 1], 
      [1, 1, 1, 0]],
    "rate_divisor": 3,
    "probability_array" : [
      [[1, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 1, 0, 0],
       [0, 1, 0, 0],
       [0, 1, 0, 0]]]
  }
}
```
