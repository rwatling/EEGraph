#include "../include/graph.hpp"
#include "../include/argument_parsing.hpp"
#include "../include/gpu_error_check.cuh"
#include "../include/cuda_includes.cuh"
#include "../include/nvmlClass.cuh"
#include "../include/gpu_utils.cuh"
#include "../include/um_virtual_graph.cuh"
#include "../include/um_graph.cuh"
#include "../include/eegraph.cuh"

#include <iostream>
#include <sys/stat.h>
#include <cstdlib>
#include <unistd.h>

int main(int argc, char** argv) {
    ArgumentParser arguments(argc, argv, true, false);

    Graph graph(arguments.input, true);
    graph.ReadGraph();

    VertexSubgraph subgraph(graph, arguments.input, true);
    subgraph.MakeSubgraph(0.05, arguments.sourceNode);

    VirtualGraph vGraph(subgraph);
	vGraph.MakeGraph();

    UMGraph um_graph(arguments.input, true);
    um_graph.ReadGraph();

    UMVertexSubgraph um_subgraph(um_graph, arguments.input, true);
    um_subgraph.MakeSubgraph(0.05, arguments.sourceNode);

    if (arguments.unifiedMem) {
        eegraph_sssp_um(arguments, um_subgraph);
    } else {
        eegraph_sssp(arguments, subgraph);
    }
}