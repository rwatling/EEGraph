#include "timer.hpp"
#include "utilities.hpp"
#include "graph.hpp"
#include "globals.hpp"
#include "argument_parsing.hpp"
#include "gpu_error_check.cuh"
#include "cuda_includes.cuh"
#include "nvmlClass.cuh"
#include "cc.cuh"
#include "virtual_graph.hpp"
#include "gpu_utils.cuh"
#include "um_virtual_graph.cuh"
#include "um_graph.cuh"
#include "bfs.cuh"
#include "cc.cuh"
#include "pr.cuh"
#include "sswp.cuh"
#include "sssp.cuh"
#include <iostream>

int eegraph_bfs(ArgumentParser arguments, Graph graph);

int eegraph_bfs_um(ArgumentParser arguments, Graph graph);

int eegraph_cc(ArgumentParser arguments, Graph graph);

int eegraph_cc_um(ArgumentParser arguments, Graph graph);

int eegraph_pr(ArgumentParser arguments, Graph graph);

int eegraph_pr_um(ArgumentParser arguments, Graph graph);

int eegraph_sswp(ArgumentParser arguments, Graph graph);

int eegraph_sswp_um(ArgumentParser arguments, Graph graph);

int eegraph_sssp(ArgumentParser arguments, Graph graph);

int eegraph_sssp_um(ArgumentParser arguments, Graph graph);