# EEGraph

### Compilation

`$ mkdir build`

`$ cmake ..`

`$ make`

### Run executable examples

`$ sssp --help`

`$ sssp --input <filename>`

### Input graph format

Input graphs should be in form of plain text files, containing the list of the edges of the graph. Each line is corresponding to an edge and is of the following form:

```
V1  V2  W
```

### Acknowledgements

The basic functionality of the code was adapted from the Tigr paper cited below

[ASPLOS'18] Amir Nodehi, Junqiao Qiu, Zhijia Zhao. Tigr: Transforming
Irregular Graphs for GPU-Friendly Graph Processing. In Proceedings of
The 23th International Conference on Architectural Support for
Programming Languages and Operating Systems, Williamsburg, VA, 2018. 15
pages
