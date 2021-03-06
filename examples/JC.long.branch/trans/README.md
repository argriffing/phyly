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
	"columns": ["site", "edge", "first_state", "second_state", "value"],
	"data": [

		[0, 0, 0, 1, 2.0416666667103849],
		[0, 0, 0, 2, 1.7916666666579231],
		[0, 0, 0, 3, 1.7916666666579231],
		[0, 0, 1, 0, 1.5416666666754104],
		[0, 0, 1, 2, 1.5416666666754104],
		[0, 0, 1, 3, 1.5416666666754104],
		[0, 0, 2, 0, 1.5416666666754104],
		[0, 0, 2, 1, 1.7916666666579231],
		[0, 0, 2, 3, 1.5416666666754104],
		[0, 0, 3, 0, 1.5416666666754104],
		[0, 0, 3, 1, 1.7916666666579231],
		[0, 0, 3, 2, 1.5416666666754104],

		[1, 0, 0, 1, 1.7916666666391241],
		[1, 0, 0, 2, 1.7916666666391241],
		[1, 0, 0, 3, 1.7916666666391241],
		[1, 0, 1, 0, 1.7916666666391241],
		[1, 0, 1, 2, 1.5416666666592345],
		[1, 0, 1, 3, 1.5416666666592345],
		[1, 0, 2, 0, 1.7916666666391241],
		[1, 0, 2, 1, 1.5416666666592345],
		[1, 0, 2, 3, 1.5416666666592345],
		[1, 0, 3, 0, 1.7916666666391241],
		[1, 0, 3, 1, 1.5416666666592345],
		[1, 0, 3, 2, 1.5416666666592345],

		[2, 0, 0, 1, 1.6041666666668306],
		[2, 0, 0, 2, 1.6041666666668306],
		[2, 0, 0, 3, 1.6041666666668306],
		[2, 0, 1, 0, 1.8541666666661749],
		[2, 0, 1, 2, 1.6041666666668306],
		[2, 0, 1, 3, 1.6041666666668306],
		[2, 0, 2, 0, 1.8541666666661749],
		[2, 0, 2, 1, 1.6041666666668306],
		[2, 0, 2, 3, 1.6041666666668306],
		[2, 0, 3, 0, 1.8541666666661749],
		[2, 0, 3, 1, 1.6041666666668306],
		[2, 0, 3, 2, 1.6041666666668306],

		[3, 0, 0, 1, 1.6666666666666667],
		[3, 0, 0, 2, 1.6666666666666667],
		[3, 0, 0, 3, 1.6666666666666667],
		[3, 0, 1, 0, 1.6666666666666667],
		[3, 0, 1, 2, 1.6666666666666667],
		[3, 0, 1, 3, 1.6666666666666667],
		[3, 0, 2, 0, 1.6666666666666667],
		[3, 0, 2, 1, 1.6666666666666667],
		[3, 0, 2, 3, 1.6666666666666667],
		[3, 0, 3, 0, 1.6666666666666667],
		[3, 0, 3, 1, 1.6666666666666667],
		[3, 0, 3, 2, 1.6666666666666667]]
}
```

Command:
```shell
$ arbplf-trans < in.json > out.json
```
