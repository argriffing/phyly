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

```shell
$ arbplf-deriv < jc29.same.json
{"columns": ["site", "edge", "value"], "data": [[0, 0, -6.4467380574161446e-17]]}
$ arbplf-deriv < jc29.diff.json 
{"columns": ["site", "edge", "value"], "data": [[0, 0, 2.1489126858053815e-17]]}
```
