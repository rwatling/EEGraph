#include "../include/timer.hpp"
#include "../include/utilities.hpp"
#include "../include/graph.hpp"
#include "../include/globals.hpp"
#include "../include/argument_parsing.hpp"
#include "../include/gpu_error_check.cuh"
#include "../include/cuda_includes.cuh"
#include "../include/nvmlClass.cuh"
#include "../include/sssp.cuh"
#include <iostream>

using namespace std;

int main(int argc, char** argv) {
	
	//Parse arguments and initialize
	ArgumentParser arguments(argc, argv, true, false);

	// Initialize graph
	Graph graph(arguments.input, true);
	graph.ReadGraph();

	uint num_nodes = graph.num_nodes;
	uint num_edges = graph.num_edges;

	if (num_nodes  < 1) {
		cout << "Graph file not read correctly" << endl;
		return -1;
	}

	// Set device
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

	// Energy structures initilization
	// Two cpu threads are used to coordinate energy consumption by chanding common flags in nvmlClass
	vector<thread> cpu_threads;
	nvmlClass nvml(arguments.deviceID, arguments.energyFile, arguments.energyStats, to_string(arguments.variant));

	if (arguments.energy) {
		cout << "Starting energy measurements. Timing information will be affected..." << endl;

		cpu_threads.emplace_back(std::thread(&nvmlClass::getStats, &nvml));

  		nvml.log_start();
	}


	// GPU variable declarations
	Edge* d_edges;
	uint* d_weights;
	unsigned int* d_dist;

	bool finished;
	bool* d_finished;
	bool finished1;
	bool* d_finished1;

	// Allocate on GPU device
	gpuErrorcheck(cudaMalloc(&d_edges, num_edges * sizeof(Edge)));
	gpuErrorcheck(cudaMalloc(&d_weights, num_edges * sizeof(uint)));
	gpuErrorcheck(cudaMalloc(&d_dist, num_nodes * sizeof(uint)));
	gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_finished1, sizeof(bool)))

	// Copy to GPU device
	gpuErrorcheck(cudaMemcpy(d_edges, graph.edges.data(), num_edges * sizeof(Edge), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_weights, graph.weights.data(), num_edges * sizeof(uint), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_dist, dist, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	
	// Algorithm control variable declarations
	Timer timer;
	uint itr = 0;
	uint num_threads = 512;
	uint edges_per_thread = 8;
	uint num_blocks = (num_edges) / (num_threads * edges_per_thread) + 1;

	timer.Start();

	//Main algorithm
	if (arguments.variant == ASYNC_PUSH_TD) {
		do {

			itr++;
			finished = true;
			gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

			sssp::async_push_td<<<num_blocks, num_threads>>>(  d_edges, 
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
	} else if (arguments.variant == SYNC_PUSH_TD) {

		do {
			itr++;

			if (itr % 2 == 0) {

				finished = true;
				gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

				sssp::sync_push_td<<<num_blocks, num_threads>>>(  d_edges, 
																d_weights, 
																num_edges, 
																edges_per_thread, 
																arguments.sourceNode, 
																d_dist,
																d_finished,
																d_finished1,
																true  );
			} else {

				finished1 = true;
				gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

				sssp::sync_push_td<<<num_blocks, num_threads>>>(  d_edges, 
																d_weights, 
																num_edges, 
																edges_per_thread, 
																arguments.sourceNode, 
																d_dist,
																d_finished,
																d_finished1,
																false  );
			}

			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck( cudaDeviceSynchronize() );
			gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
			gpuErrorcheck(cudaMemcpy(&finished1, d_finished1, sizeof(bool), cudaMemcpyDeviceToHost));

		} while (!finished && !finished1);

	}

	// Copy back to host
	gpuErrorcheck(cudaMemcpy(dist, d_dist, num_nodes*sizeof(unsigned int), cudaMemcpyDeviceToHost));

	cout << "Number of iterations = " << itr << endl;

	float runtime = timer.Finish();
	cout << "Processing finished in " << runtime << " (ms).\n";

	// Stop measuring energy consumption, clean up structures
	if (arguments.energy) {
		cpu_threads.emplace_back(thread( &nvmlClass::killThread, &nvml));

		for (auto& th : cpu_threads) {
			th.join();
			th.~thread();
		}

		cpu_threads.clear();
	}

	// Run sequential cpu version and print out useful information
	if (arguments.debug) {
		unsigned int* cpu_dist;
		cpu_dist = new unsigned int[num_nodes];

		for(int i=0; i<num_nodes; i++)
		{
			cpu_dist[i] = DIST_INFINITY;
		}
		
		cpu_dist[arguments.sourceNode] = 0;

		sssp::seq_cpu(	graph.edges, 
					    graph.weights, 
					    num_edges, 
					    arguments.sourceNode, 
					    cpu_dist);

		if (num_nodes < 20) {
			utilities::PrintResults(cpu_dist, num_nodes);
			utilities::PrintResults(dist, num_nodes);
		} else {
			utilities::PrintResults(cpu_dist, 10);
			utilities::PrintResults(dist, 10);
		}

		utilities::CompareArrays(cpu_dist, dist, num_nodes);
	}


	gpuErrorcheck(cudaFree(d_edges));
	gpuErrorcheck(cudaFree(d_weights));
	gpuErrorcheck(cudaFree(d_dist));
	gpuErrorcheck(cudaFree(d_finished));
}
