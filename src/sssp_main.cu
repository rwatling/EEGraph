#include "../include/timer.hpp"
#include "../include/tigr_utilities.hpp"
#include "../include/graph.hpp"
#include "../include/globals.hpp"
#include "../include/argument_parsing.hpp"
#include "../include/gpu_error_check.cuh"
#include "../include/cuda_includes.cuh"
#include "../include/nvmlClass.cuh"
#include <iostream>

void sssp_cpu(vector<Edge> edges, vector<uint> weights, uint num_nodes, uint num_edges, int source, int itrs, unsigned int* dist) {

	bool finished = false;
	int iterations;

	if (itrs < 1) {
		itrs = num_nodes - 1;
	}

	while (!finished) {
		finished = true;

		Edge e;
		uint e_w8;

		for (int i = 0; i < num_edges; i++) {
			e = edges.at(i);
			e_w8 = weights.at(i);

			if (dist[e.source] + e_w8 < dist[e.end]) {
				dist[e.end] = dist[e.source] + e_w8;
				finished = false;
			}
		}

		iterations++;

		if(iterations >= itrs) {
			finished = true;
		}

	}

}

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
	
	for(int i=0; i<num_nodes; i++)
	{
			dist[i] = DIST_INFINITY;
	}
	
	dist[arguments.sourceNode] = 0;

	sssp_cpu(graph.edges, graph.weights, num_nodes, num_edges, 0, 0, dist);

	utilities::PrintResults(dist, 30);
}
