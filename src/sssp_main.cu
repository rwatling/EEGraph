#include "../include/timer.hpp"
#include "../include/tigr_utilities.hpp"
#include "../include/graph.hpp"
#include "../include/globals.hpp"
#include "../include/argument_parsing.hpp"
#include "../include/gpu_error_check.cuh"
#include "../include/cuda_includes.cuh"
#include "../include/nvmlClass.cuh"
#include <iostream>

int main(int argc, char** argv) {
	
	ArgumentParser arguments(argc, argv, true, false);

	Graph graph(arguments.input, true);
	graph.ReadGraph();

	uint num_nodes = graph.num_nodes;
	uint num_edges = graph.num_edges;

	if(arguments.hasDeviceID)
		cudaSetDevice(arguments.deviceID);

	unsigned int *dist;
	dist = new unsigned int[num_nodes];

	bool *label1;
	bool *label2;
	label1 = new bool[num_nodes];
	label2 = new bool[num_nodes];
	
	for(int i=0; i<num_nodes; i++)
	{
			dist[i] = DIST_INFINITY;
			label1[i] = false;
			label2[i] = false;
	}
	
	dist[arguments.sourceNode] = 0;
	label1[arguments.sourceNode] = true;
}
