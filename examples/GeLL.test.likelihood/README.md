This example uses a general time-reversible model of DNA evolution
on a small evolutionary tree,
with discretized gamma distributed rates across sites.
Of the several `arbplf-*` functions,
only the log likelihood is demonstrated here.

The example has been taken from the
[likelihood test](https://github.com/danielmoney/GeLL/blob/master/test/Likelihood/LikelihoodTest.java)
in the GeLL suite of unit tests.

Log likelihood computed by GeLL:
```
-2616.0738920000026
```

Log likelihood expected by the GeLL test suite,
presumably obtained from PAML:
```
-2616.073763
```

The log likelihood is a function of the `brown.nuc` alignment,
and of some hard-coded values in the test suite.

These various quantities have been merged into an `in.json` file,
and `arbplf-ll` has been used to reproduce the log likelihood below.

```shell
$ arbplf-ll < in.json
{"columns": ["value"], "data": [[-2616.073919844292]]}
```

As expected, the log likelihoods are pretty similar.
