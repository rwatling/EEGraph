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

Result eegraph_bfs(ArgumentParser &arguments, Graph &graph);

Result eegraph_bfs_um(ArgumentParser &arguments, UMGraph &graph);

Result eegraph_cc(ArgumentParser &arguments, Graph &graph);

Result eegraph_cc_um(ArgumentParser &arguments, UMGraph &graph);

Result eegraph_pr(ArgumentParser &arguments, Graph &graph);

Result eegraph_pr_um(ArgumentParser &arguments, UMGraph &graph);

Result eegraph_sswp(ArgumentParser &arguments, Graph &graph);

Result eegraph_sswp_um(ArgumentParser &arguments, UMGraph &graph);

Result eegraph_sssp(ArgumentParser &arguments, Graph &graph);

Result eegraph_sssp_um(ArgumentParser &arguments, UMGraph &graph);