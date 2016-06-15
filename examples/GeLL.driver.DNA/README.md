This example uses a general time-reversible model of DNA evolution
on a small evolutionary tree,
with discretized gamma distributed rates across sites.
Of the several `arbplf-*` functions,
only log likelihood is demonstrated here.

The example has been taken from the maximum likelihood
parameter estimates found by GeLL in its DNA 'driver'
[example](http://phylo.bio.ku.edu/GeLL/DriverExamples.tar.gz).


```shell
GeLL/DNA$ java -jar ../GeLL.jar settings.dat
```

Log likelihood:
```
-2616.0735599559825
```

This log likelihood is a function of the `brown.nuc` alignment,
and of the tree shape, branch lengths, and the GTR and gamma rate parameter
estimates found by GeLL and recorded in its
`outtree.dat` and `outparameters.dat` output files.

These various quantities have been merged into an `in.json` file,
and `arbplf-ll` has been used to reproduce the log likelihood below.

```shell
$ arbplf-ll < in.json
{"columns": ["value"], "data": [[-2616.0735881244163]]}
```

As expected, the log likelihoods are pretty similar.
