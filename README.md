# EEGraph

### Compilation

`$ mkdir build`

`$ cmake ..`

`$ make`

### Run executable

`$ sssp --help`

`$ sssp --input <filename>`

### Input graph format

Input graphs should be in form of plain text files, containing the list of the edges of the graph. Each line is corresponding to an edge and is of the following form:

```
V1  V2  W
```

### Dataset sources

#### [Network Repository](https://networkrepository.com/)
* kron
* road_usa
* soc-LiveJournal1
* soc-orkut
* soc-twitter-2010

#### [Stanford SNAP Datasets](https://snap.stanford.edu/data/index.html)
* twitter-ego
* facebook_combined

#### [Galois Project](https://github.com/IntelligentSoftwareSystems/Galois)
* Build the graph convert tool to convert and use to convert from a graph type to edgelist
* [SOSP '13] Donald Nguyen, Andrew Lenharth, and Keshav Pingali. A Light-
weight Infrastructure for Graph Analytics. In Proceedings of the Twenty-
Fourth ACM Symposium on Operating Systems Principles, 2013.
ACM, New York, NY, USA, 456–471


### Acknowledgements

The basic functionality of the code was adapted from the Tigr paper cited below

[ASPLOS'18] Amir Nodehi, Junqiao Qiu, Zhijia Zhao. Tigr: Transforming
Irregular Graphs for GPU-Friendly Graph Processing. In Proceedings of
The 23th International Conference on Architectural Support for
Programming Languages and Operating Systems, Williamsburg, VA, 2018. 15
pages