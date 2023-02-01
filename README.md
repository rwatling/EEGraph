# EEGraph

### Compilation

`$ mkdir build`

`$ cd build`

`$ cmake ..`

`$ make`

### Versions

* nvcc 11.1
* gcc 7.3.1
* cmake 3.20.0-rc4
* C++11 multithreading
* C++14

### Run executable examples

`$ sssp --help`

`$ sssp --input <filename>`

### Input graph format

Input graphs should be in form of plain text files, containing the list of the edges of the graph. Each line is corresponding to an edge and is of the following form:

```
V1  V2
```

### Acknowledgements

The basic functionality of the code was adapted from the Tigr and Subway

[EUROSYS'20] Amir Hossein Nodehi Sabet, Zhijia Zhao, and Rajiv Gupta. [Subway: minimizing data transfer during out-of-GPU-memory graph processing](https://dl.acm.org/doi/abs/10.1145/3342195.3387537). In Proceedings of the Fifteenth European Conference on Computer Systems.

[ASPLOS'18] Amir Hossein Nodehi Sabet, Junqiao Qiu, and Zhijia Zhao. [Tigr: Transforming Irregular Graphs for GPU-Friendly Graph Processing](https://dl.acm.org/doi/10.1145/3173162.3173180). In Proceedings of the Twenty-Third International Conference on Architectural Support for Programming Languages and Operating Systems.

Additionally, nvmlClass.h was modified from the following repository
https://github.com/mnicely/nvml_examples