Command:
```shell
$ arbplf-marginal < in.json > out.json
```

Output (out.json):
```json
{
  "columns": ["site", "node", "state", "value"],
  "data": [

    [0, 0, 0, 1.0],
    [0, 0, 1, 0.0],
    [0, 0, 2, 0.0],
    [0, 0, 3, 0.0],

    [0, 1, 0, 0.0],
    [0, 1, 1, 1.0],
    [0, 1, 2, 0.0],
    [0, 1, 3, 0.0],

    [0, 2, 0, 0.0],
    [0, 2, 1, 1.0],
    [0, 2, 2, 0.0],
    [0, 2, 3, 0.0],

    [0, 3, 0, 0.11955834531728772],
    [0, 3, 1, 0.65793082312499807],
    [0, 3, 2, 0.11125541577885713],
    [0, 3, 3, 0.11125541577885713],

    [0, 4, 0, 0.2608466924812069],
    [0, 4, 1, 0.33744518055975692],
    [0, 4, 2, 0.20085406347951812],
    [0, 4, 3, 0.20085406347951812]]
}
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
       [1, 1, 1, 1],
       [1, 1, 1, 1]]]
  }
}
```