GeLL: /test/Likelihood/LikelihoodTest.java

GeLL log likelihood (I had to add a System.out.println to see this):

-2616.0738920000026

The log likelihood that GeLL compares against for testing,
presumably obtained from PAML:

-2616.073763

Log likelihood reproduced using arbplf-ll:
```shell
$ python mknuc.py > nuc.json
$ python mkmodel.py > model.json
$ jq -s '.[0] * .[1]' nuc.json model.json > in.json
$ arbplf-ll < in.json
{"columns": ["value"], "data": [[-2616.073919844292]]}
```
