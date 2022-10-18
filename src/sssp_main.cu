#include "../include/timer.hpp"
#include "../include/utilities.hpp"
#include "../include/graph.hpp"
#include "../include/globals.hpp"
#include "../include/argument_parsing.hpp"
#include "../include/gpu_error_check.cuh"
#include "../include/cuda_includes.cuh"
#include "../include/nvmlClass.cuh"
#include <iostream>

__global__ void sssp_gpu(	Edge* edges, 
							uint* weights, 
							uint num_edges,
							uint edges_per_thread, 
							int source,
							unsigned int* dist,
							bool* finished		) {

	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int startId = threadId * edges_per_thread;
    
    if (startId >= num_edges) {
        return;
    }
    
    int endId = (threadId + 1) * edges_per_thread;
    if (endId >= num_edges) {
        endId = num_edges;
    }

	Edge e;
	uint e_w8;
	uint final_dist;

	for (int partId = startId; partId < endId; partId++) {
		e = edges[partId];
		e_w8 = weights[partId];
		final_dist = dist[e.source] + e_w8;

		if (final_dist < dist[e.end]) {
			atomicMin(&dist[e.end], final_dist);
			*finished = false;
		}
	}
}

void sssp_cpu(	vector<Edge> edges, 
				vector<uint> weights, 
				uint num_edges, 
				int source, 
				unsigned int* dist	) {

	bool finished = false;

	while (!finished) {
		finished = true;

		Edge e;
		uint e_w8;
		uint final_dist;

		for (int i = 0; i < num_edges; i++) {
			e = edges.at(i);
			e_w8 = weights.at(i);
			final_dist = dist[e.source] + e_w8;

			if (final_dist < dist[e.end]) {
				dist[e.end] = final_dist;
				finished = false;
			}
		}
	}
}

int main(int argc, char** argv) {
	
	//Parse arguments and initialize
	ArgumentParser arguments(argc, argv, true, false);

	Graph graph(arguments.input, true);
	graph.ReadGraph();

	uint num_nodes = graph.num_nodes;
	uint num_edges = graph.num_edges;

	if(arguments.hasDeviceID)
		gpuErrorcheck(cudaSetDevice(arguments.deviceID));

	// Distance initialization
	unsigned int *dist;
	dist = new unsigned int[num_nodes];

	for(int i=0; i<num_nodes; i++)
	{
		dist[i] = DIST_INFINITY;
	}
	
	dist[arguments.sourceNode] = 0;

	// GPU variable declarations
	Edge* d_edges;
	uint* d_weights;
	unsigned int* d_dist;

	bool finished;
	bool* d_finished;

	// Allocate on GPU device
	gpuErrorcheck(cudaMalloc(&d_edges, num_edges * sizeof(Edge)));
	gpuErrorcheck(cudaMalloc(&d_weights, num_edges * sizeof(uint)));
	gpuErrorcheck(cudaMalloc(&d_dist, num_nodes * sizeof(uint)));
	gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));

	// Copy to GPU device
	gpuErrorcheck(cudaMemcpy(d_edges, graph.edges.data(), num_edges * sizeof(Edge), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_weights, graph.weights.data(), num_edges * sizeof(uint), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_dist, dist, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	
	// Algorithm control variable declarations
	Timer timer;
	uint itr = 0;
	uint num_threads = 512;
	uint edges_per_thread = num_edges / num_threads + 1;
	uint num_blocks = num_nodes / num_threads;

	timer.Start();

	//Main algorithm
	do {
		itr++;
		finished = true;
		gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

		sssp_gpu<<<num_blocks, num_threads>>>(	d_edges, 
												d_weights, 
												num_edges, 
												edges_per_thread, 
												arguments.sourceNode, 
												d_dist, 
												d_finished	);

		gpuErrorcheck( cudaPeekAtLastError() );
		gpuErrorcheck( cudaDeviceSynchronize() );	
		
		gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
	} while (!(finished));

	cout << "Number of iterations = " << itr << endl;

	float runtime = timer.Finish();
	cout << "Processing finished in " << runtime << " (ms).\n";

	gpuErrorcheck(cudaMemcpy(dist, d_dist, num_nodes*sizeof(unsigned int), cudaMemcpyDeviceToHost));

	if (arguments.debug) {
		unsigned int* cpu_dist;
		cpu_dist = new unsigned int[num_nodes];

		for(int i=0; i<num_nodes; i++)
		{
			cpu_dist[i] = DIST_INFINITY;
		}
		
		cpu_dist[arguments.sourceNode] = 0;

		sssp_cpu(	graph.edges, 
					graph.weights, 
					num_edges, 
					arguments.sourceNode, 
					cpu_dist);

		if (num_nodes < 20) {
			utilities::PrintResults(cpu_dist, num_nodes);
			utilities::PrintResults(dist, num_nodes);
		} else {
			utilities::PrintResults(cpu_dist, 20);
			utilities::PrintResults(dist, 20);
		}

		utilities::CompareArrays(cpu_dist, dist, num_nodes);
	}


	gpuErrorcheck(cudaFree(d_edges));
	gpuErrorcheck(cudaFree(d_weights));
	gpuErrorcheck(cudaFree(d_dist));
	gpuErrorcheck(cudaFree(d_finished));
}
