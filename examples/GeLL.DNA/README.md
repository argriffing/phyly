GeLL 'driver' DNA example
---

GeLL settings and computed log likelihood:
---

DNA/settings.dat:
```
[Likelihood]
AlignmentType	Sequence
Alignment		data/brown.nuc
TreeInput		data/tree.dat
Model			data/model.dat
ParameterInput	data/parameters.dat
TreeOutput		outtree.dat
ParameterOutput	outparameters.dat

[Ancestral]
Type			Joint
Output			ancestor.dat

[Simulate]
Length			500
Output			simulated.dat
```

DNA/outtree.dat
```
((Gorilla:0.07435276303709204,(Chimpanzee:0.07462376514241155,Human:0.057990312243662766)A:0.0355060511914654)B:0.13146876021580028,Gibbon:0.5448816311811152,Orangutan:0.350348976027984)C;
```

DNA/outparameters.dat:
(The parameters `{a, b, c, d, e}` define the symmetric part of a GTR
rate matrix, and the `g` parameter defines the shape parameter of a 4-category
discretized gamma rate mixture).
```
Like	-2616.0735599559825

Chimpanzee	0.07462376514241155
Human	0.057990312243662766
Gorilla	0.07435276303709204
A	0.0355060511914654
Gibbon	0.5448816311811152
Orangutan	0.350348976027984
B	0.13146876021580028
a	0.8671005148903497
b	0.026470048816886478
c	0.0
d	0.049264783448354395
e	0.011075451195349906
g	0.19242344607262146
pT	1.0
pC	1.471797772941626
pA	1.3744425342086881
pG	0.40904476825485114
```

An attempt to reproduce this log likelihood using arbplf-ll:
---

```shell
$ python mknuc.py > nuc.json
$ python mkmodel.py > model.json
$ jq -s '.[0] * .[1]' nuc.json model.json > in.json
$ arbplf-ll < in.json
{"columns": ["value"], "data": [[-2616.073588220348]]}
```
