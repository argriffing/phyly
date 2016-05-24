Here we compute the log likelihood for the example in Figure 16.4 of
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
   }
}
```

Commands:
```bash
$ arbplf-ll < in.json > out.json
$ tabby < out.json > out.md
```

Output (out.json):
```json
{
  "columns": ["site", "value"],
  "data": [[0, -11.297288182875496]]
}
```

Output rendered as a table (out.md):

|    |   site |               value |
|---:|-------:|--------------------:|
|  0 |      0 | -11.297288182875496 |

Converting from log likelihood to likelihood:
```shell
$ echo 'e(-11.297288182875496)' | bc -l
.00001240652290578037
```

---

This second example computes log likelihoods for three sites
of an alignment. For the first site, no data is available and therefore
the associated likelihood is 1 and the log likelihood is 0.
The second site is the same as the example above; data is available
only at the leaves of the tree. At the third site we assume that
sequence data is available for all taxa, including internal nodes.
The likelihood computed for the third site agrees with the likelihood
reported by Felsenstein.

Input (in2.json):
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
	  [1, 1, 1, 1]],
	 [[1, 0, 0, 0],
	  [0, 1, 0, 0],
	  [0, 1, 0, 0],
	  [0, 1, 0, 0],
	  [0, 0, 1, 0],
	  [1, 1, 1, 1],
	  [1, 1, 1, 1],
	  [1, 1, 1, 1]],
	 [[1, 0, 0, 0],
	  [0, 1, 0, 0],
	  [0, 1, 0, 0],
	  [0, 1, 0, 0],
	  [0, 0, 1, 0],
	  [0, 0, 1, 0],
	  [0, 0, 1, 0],
	  [1, 0, 0, 0]]]
   }
}
```

Commands:
```bash
$ arbplf-ll < in2.json | tabby > out2.md
```

Output rendered as a table (out2.md):

|    |   site |               value |
|---:|-------:|--------------------:|
|  0 |      0 |   0                 |
|  1 |      1 | -11.297288182875496 |
|  2 |      2 | -12.390132492111672 |

Converting from log likelihood to likelihood:
```shell
$ echo 'e(-12.390132492111672)' | bc -l
.00000415943008401296
```
