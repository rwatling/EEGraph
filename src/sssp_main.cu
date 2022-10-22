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

	if(arguments.hasDeviceID)
		gpuErrorcheck(cudaSetDevice(arguments.deviceID));

	// Distance and active initialization
	unsigned int *dist;
	bool* active_current;
	bool* active_next;

	dist = new unsigned int[num_nodes];
	active_current = new bool[num_nodes];
	active_next = new bool[num_nodes];

	for(int i=0; i<num_nodes; i++)
	{
		dist[i] = DIST_INFINITY;
		active_current[i] = false;
		active_next[i] = false;
	}
	
	dist[arguments.sourceNode] = 0;
	active_current[arguments.sourceNode] = true;
	active_next[arguments.sourceNode] = true;

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
	bool* d_active_current;
	bool* d_active_next;
	bool* d_finished;

	// Allocate on GPU device
	gpuErrorcheck(cudaMalloc(&d_edges, num_edges * sizeof(Edge)));
	gpuErrorcheck(cudaMalloc(&d_weights, num_edges * sizeof(uint)));
	gpuErrorcheck(cudaMalloc(&d_dist, num_nodes * sizeof(uint)));
	gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_active_current, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_active_next, num_nodes * sizeof(bool)));

	// Copy to GPU device
	gpuErrorcheck(cudaMemcpy(d_edges, graph.edges.data(), num_edges * sizeof(Edge), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_weights, graph.weights.data(), num_edges * sizeof(uint), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_dist, dist, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_active_current, active_current, num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_active_next, active_next, num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	
	// Algorithm control variable declarations
	Timer timer;
	uint itr = 0;
	uint num_threads = 512;
	uint edges_per_thread = 8;
	uint nodes_per_thread = 8;
	uint num_blocks_edgelist = (num_edges) / (num_threads * edges_per_thread) + 1;
	uint num_blocks_node_centric = (num_nodes) / (num_threads * nodes_per_thread) + 1;
	bool finished;

	timer.Start();

	//Main algorithm
	/*if (arguments.variant == ASYNC_PUSH_TD) {
		do {

			itr++;
			finished = true;
			gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

			sssp::async_push_td<<<num_blocks_edgelist, num_threads>>>(  d_edges, 
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
			finished = true;
			gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

			if (itr % 2 == 0) {
				
				sssp::sync_push_td<<<num_blocks_edgelist, num_threads>>>(  d_edges, 
																d_weights, 
																num_edges, 
																edges_per_thread, 
																arguments.sourceNode, 
																d_dist,
																d_finished,
																true  );
			} else {
				sssp::sync_push_td<<<num_blocks_edgelist, num_threads>>>(  d_edges, 
																d_weights, 
																num_edges, 
																edges_per_thread, 
																arguments.sourceNode, 
																d_dist,
																d_finished,
																false  );
			}

			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck( cudaDeviceSynchronize() );
			gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));

		} while (!finished);

	} else if (arguments.variant == SYNC_PUSH_DD) {

		do {
			itr++;
			finished = true;
			gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));
			
			if (itr % 2 == 0) {


				sssp::sync_dd_clear_active<<<num_blocks_node_centric, num_threads>>>(d_active_next, num_nodes, nodes_per_thread);

				sssp::sync_push_dd<<<num_blocks_edgelist, num_threads>>>(  d_edges, 
																  d_weights, 
																  num_edges, 
																  edges_per_thread, 
																  arguments.sourceNode, 
																  d_dist,
																  d_finished,
																  d_active_current,
																  d_active_next,
																  true  );
			} else {
				sssp::sync_dd_clear_active<<<num_blocks_node_centric, num_threads>>>(d_active_current, num_nodes, nodes_per_thread);

				sssp::sync_push_dd<<<num_blocks_node_centric, num_threads>>>(  d_edges, 
																  d_weights, 
																  num_edges, 
																  edges_per_thread, 
																  arguments.sourceNode, 
																  d_dist,
																  d_finished,
																  d_active_next,
																  d_active_current,
																  false  );

			}

			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck( cudaDeviceSynchronize() );
			gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
			gpuErrorcheck(cudaMemcpy(active_current, d_active_current, num_nodes * sizeof(bool), cudaMemcpyDeviceToHost));
			gpuErrorcheck(cudaMemcpy(active_next, d_active_next, num_nodes * sizeof(bool), cudaMemcpyDeviceToHost));

		} while (!finished);
	}*/

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
			utilities::PrintResults(cpu_dist, 20);
			utilities::PrintResults(dist, 20);
		}

		utilities::CompareArrays(cpu_dist, dist, num_nodes);
	}


	gpuErrorcheck(cudaFree(d_edges));
	gpuErrorcheck(cudaFree(d_weights));
	gpuErrorcheck(cudaFree(d_dist));
	gpuErrorcheck(cudaFree(d_finished));
	gpuErrorcheck(cudaFree(d_active_current));
	gpuErrorcheck(cudaFree(d_active_next));
}
