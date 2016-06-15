Input (in.json):
```json
{
  "model_and_data" : {
    "edges" : [[0, 1]],
    "edge_rate_coefficients" : [20],
    "root_prior" : [0.25, 0.25, 0.25, 0.25],
    "rate_matrix" : [
      [0, 1, 1, 1],
      [1, 0, 1, 1],
      [1, 1, 0, 1],
      [1, 1, 1, 0]],
    "rate_divisor" : 3,
    "character_definitions" : [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [1, 1, 1, 1]],
    "character_data" : [
      [0, 1],
      [0, 0],
      [2, 0],
      [2, 2]]
  }
}
```

Output (out.json):
```json
{
  "columns": ["site", "edge", "value"],
  "data": [
    [0, 0, 3.4974583595682378e-12],
    [1, 0, -1.0492375078594624e-11],
    [2, 0, 0.0],
    [3, 0, 0.0]]
}
```

Command:
```shell
$ arbplf-deriv < in.json > out.json
```
