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
  "columns": ["site", "value"],
  "data": [
    [0, -2.7725887222424044],
    [1, -2.7725887222319119],
    [2, -1.3862943611198906],
    [3, 0.0]]
}
```

Command:
```shell
$ arbplf-ll < in.json > out.json
```
