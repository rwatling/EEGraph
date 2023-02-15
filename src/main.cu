#include "../include/timer.hpp"
#include "../include/utilities.hpp"
#include "../include/graph.hpp"
#include "../include/globals.hpp"
#include "../include/argument_parsing.hpp"
#include "../include/gpu_error_check.cuh"
#include "../include/cuda_includes.cuh"
#include "../include/nvmlClass.cuh"
#include "../include/sssp.cuh"
#include "../include/virtual_graph.hpp"
#include "../include/gpu_utils.cuh"
#include "../include/um_virtual_graph.cuh"
#include "../include/um_graph.cuh"

#include <iostream>

int main (int argc, char** argv) {
    ArgumentParser arguments(argc, argv, true, false);
    
    // Initialize graph
	Graph graph(arguments.input, true);
	graph.ReadGraph();
}