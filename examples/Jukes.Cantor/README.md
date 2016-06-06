For a single branch of a Jukes-Cantor process,
the double-precision log likelihood is indistinguishable from
`-log(4)` when the branch length is large enough that at least
29 substitutions are expected on the branch.
This is true regardless of whether the initial and final states
are identical or non-identical.

```shell
$ arbplf-ll < jc29.same.json 
{"columns": ["site", "value"], "data": [[0, -1.3862943611198906]]}
$ arbplf-ll < jc29.diff.json 
{"columns": ["site", "value"], "data": [[0, -1.3862943611198906]]}
$ arbplf-ll < jc30.same.json 
{"columns": ["site", "value"], "data": [[0, -1.3862943611198906]]}
$ arbplf-ll < jc30.diff.json 
{"columns": ["site", "value"], "data": [[0, -1.3862943611198906]]}
```

On the other hand, the derivative of the log likelihood with respect
to the branch length is informative even when the expected number
of substitutions is relatively large.

```shell
$ arbplf-deriv < jc29.same.json
{"columns": ["site", "edge", "value"], "data": [[0, 0, -6.4467380574161446e-17]]}
$ arbplf-deriv < jc29.diff.json 
{"columns": ["site", "edge", "value"], "data": [[0, 0, 2.1489126858053815e-17]]}
```

These derivatives can be checked using explicit formulas.

```python
>>> -4 / (exp(4*29 / 3) + 3)
-6.4467380574161606e-17
>>> 4 / (3*exp(4*29 / 3) - 3)
2.1489126858053865e-17
```

However if the branch lengths are large enough,
it is possible that the derivative of the log likelihood with
respect to the branch length is closer to zero than to any
number representeable in double precision.
Because the numbers in the JSON output are double-precision approximations,
the result in this case is uninformative even if it is computed without
precision loss due to internal rounding or cancellation.
This can occur when the derivatives are smaller than around 10^-300.

```shell
$ arbplf-deriv < jc600.same.json 
{"columns": ["site", "edge", "value"], "data": [[0, 0, 0.0]]}
```
