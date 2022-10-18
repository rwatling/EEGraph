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

### Dataset sources

#### [Network Repository](https://networkrepository.com/)
* [kron-g500-log21](https://networkrepository.com/kron-g500-logn21.php)
* [road_usa](https://networkrepository.com/road-usa.php)
* [soc-LiveJournal1](https://networkrepository.com/soc-LiveJournal1.php)
* [soc-orkut](https://networkrepository.com/soc-orkut.php)
* [soc-twitter-2010](https://networkrepository.com/soc-twitter-2010.php)
* Alternatively you can download these datasets from [SEP-Graph](https://github.com/SEP-Graph/ppopp19-artifact)

#### [Stanford SNAP Datasets](https://snap.stanford.edu/data/index.html)
* [twitter-ego](https://snap.stanford.edu/data/ego-Twitter.html) --included--
* [facebook_combined](https://snap.stanford.edu/data/ego-Facebook.html) --included--

#### Other
* test.txt (small test case)

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
