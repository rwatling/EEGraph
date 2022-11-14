#include "../include/timer.hpp"
#include "../include/utilities.hpp"
#include "../include/graph.hpp"
#include "../include/globals.hpp"
#include "../include/argument_parsing.hpp"
#include "../include/gpu_error_check.cuh"
#include "../include/cuda_includes.cuh"
#include "../include/nvmlClass.cuh"
#include "../include/bfs.cuh"
#include "../include/virtual_graph.hpp"
#include "../include/gpu_utils.cuh"
#include <iostream>

int main(int argc, char** argv) {

	ArgumentParser arguments(argc, argv, true, false);

	/*if (arguments.unifiedMem) {
		main_unified_memory(arguments);
	} else if (arguments.subway) {
		main_subway(arguments);
		cout << "Subway not yet implemented" << endl;
	}*/

	// Energy structures initilization
	// Two cpu threads are used to coordinate energy consumption by chanding common flags in nvmlClass
	vector<thread> cpu_threads;
	nvmlClass nvml(arguments.deviceID, arguments.energyFile, arguments.energyStats, to_string(arguments.variant));

	if (arguments.energy) {
		cout << "Starting energy measurements. Timing information will be affected..." << endl;

		cpu_threads.emplace_back(std::thread(&nvmlClass::getStats, &nvml));

  		nvml.log_start();
	}

	// Initialize graph and virtual graph
	Graph graph(arguments.input, true);
	
	if ((graph.getFileExtension(graph.graphFilePath) == "bcsr") || (graph.getFileExtension(graph.graphFilePath) == "bwcsr")) {
		cout << "bcsr and bwcsr files are inteded to run on um or subway only" << endl;
		exit(0);
	}

	graph.ReadGraph();

	VirtualGraph vGraph(graph);
	vGraph.MakeGraph();

	/*if (!bfs::checkSize(graph, vGraph, arguments.deviceID)) {
		cout << "Graph too large! Switching to unified memory" << endl;
		main_unified_memory(arguments);
	}*/

	uint num_nodes = graph.num_nodes;
	uint num_edges = graph.num_edges;

	if (num_nodes  < 1) {
		cout << "Graph file not read correctly" << endl;
		return -1;
	}

	if(arguments.hasDeviceID)
		gpuErrorcheck(cudaSetDevice(arguments.deviceID));

	cudaFree(0);

	unsigned int *dist;
	dist  = new unsigned int[num_nodes];

	bool *label1;
	bool *label2;
	label1 = new bool[num_nodes];
	label2 = new bool[num_nodes];
	
	for(int i=0; i<num_nodes; i++)
	{
			dist[i] = DIST_INFINITY;
			label1[i] = true;
			label2[i] = false;
	}
	
	dist[arguments.sourceNode] = 0;

	uint *d_nodePointer;
	uint *d_edgeList;
	uint *d_dist;
	PartPointer *d_partNodePointer; 
	bool *d_label1;
	bool *d_label2;
	
	bool finished;
	bool finished2;
	bool *d_finished;
	bool *d_finished2;

	if (arguments.energy) nvml.log_point();

	gpuErrorcheck(cudaMalloc(&d_nodePointer, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_edgeList, (2*num_edges + num_nodes) * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_dist, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_finished2, sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label1, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label2, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_partNodePointer, vGraph.numParts * sizeof(PartPointer)));

	gpuErrorcheck(cudaMemcpy(d_nodePointer, vGraph.nodePointer, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_edgeList, vGraph.edgeList, (2*num_edges + num_nodes) * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_dist, dist, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_label1, label1, num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_label2, label2, num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_partNodePointer, vGraph.partNodePointer, vGraph.numParts * sizeof(PartPointer), cudaMemcpyHostToDevice));

	if (arguments.energy) nvml.log_point();

	// Algorithm control variable declarations
	Timer timer;
	int itr = 0;
	unsigned int level = 0;
	uint num_threads = 512;
	uint num_blocks = vGraph.numParts / num_threads + 1;

	timer.Start();

	if (arguments.variant == SYNC_PUSH_DD) {
		do
		{
			itr++;
			finished = true;
			gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));
			
			if(itr % 2 == 1)
			{

				bfs::sync_push_dd<<< num_blocks , num_threads >>>(vGraph.numParts, 
															d_nodePointer,
															d_partNodePointer,
															d_edgeList, 
															d_dist, 
															d_finished,
															d_label1,
															d_label2,
															level);

				moveUpLabels<<< num_blocks , num_threads >>>(d_label2, d_label1, num_nodes);
			}
			else
			{
				bfs::sync_push_dd<<< num_blocks , num_threads >>>(vGraph.numParts, 
															d_nodePointer, 
															d_partNodePointer,
															d_edgeList, 
															d_dist, 
															d_finished,
															d_label2,
															d_label1,
															level);
				
				moveUpLabels<<< num_blocks , num_threads >>>(d_label1, d_label2, num_nodes);
			}

			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck( cudaDeviceSynchronize() );	
			
			gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));

			level++;

		} while (!(finished));
	} else if (arguments.variant == ASYNC_PUSH_TD) {
		do
		{
			itr++;
			finished = true;
			gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

			bfs::async_push_td<<< num_blocks , num_threads >>>(vGraph.numParts, 
														d_nodePointer,
														d_partNodePointer,
														d_edgeList, 
														d_dist, 
														d_finished,
														level);

			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck( cudaDeviceSynchronize() );	
			
			gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
			
			level++;

		} while (!(finished));
	} else if (arguments.variant == SYNC_PUSH_TD) {
		
		unsigned int level2 = 0;
		
		do
		{
			itr++;
			if(itr % 2 == 1)
			{
				finished = true;
				gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

				bfs::sync_push_td<<< num_blocks , num_threads >>>(vGraph.numParts, 
															d_nodePointer,
															d_partNodePointer,
															d_edgeList, 
															d_dist, 
															d_finished,
															false,
															level);

				level++;
			}
			else
			{
				finished2 = true;
				gpuErrorcheck(cudaMemcpy(d_finished2, &finished2, sizeof(bool), cudaMemcpyHostToDevice));
				bfs::sync_push_td<<< num_blocks , num_threads >>>(vGraph.numParts, 
															d_nodePointer, 
															d_partNodePointer,
															d_edgeList, 
															d_dist, 
															d_finished2,
															true,
															level2);
				level2++;
			}

			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck( cudaDeviceSynchronize() );	
			
			gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
			gpuErrorcheck(cudaMemcpy(&finished2, d_finished2, sizeof(bool), cudaMemcpyDeviceToHost));

		} while (!(finished) && !(finished2));
	} else if (arguments.variant == ASYNC_PUSH_DD) {
		do
		{
			itr++;
			finished = true;
			gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

			bfs::async_push_dd<<< num_blocks , num_threads >>>(vGraph.numParts, 
														d_nodePointer,
														d_partNodePointer,
														d_edgeList, 
														d_dist, 
														d_finished,
														(itr%2==1) ? d_label1 : d_label2,
														(itr%2==1) ? d_label2 : d_label1,
														level);
			mixLabels<<<num_blocks, num_threads>>>((itr%2==1)? d_label1 : d_label2, (itr%2==1)? d_label2 : d_label1, num_nodes);

			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck( cudaDeviceSynchronize() );	
			
			gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
			
			level++;

		} while (!(finished));
	}

	gpuErrorcheck(cudaMemcpy(dist, d_dist, num_nodes*sizeof(unsigned int), cudaMemcpyDeviceToHost));

	if (arguments.energy) nvml.log_point();

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

		bfs::seq_cpu(	graph.edges, 
					    graph.weights, 
					    num_edges, 
					    arguments.sourceNode, 
					    cpu_dist);

		utilities::PrintResults(cpu_dist, min(30, num_nodes));
		utilities::PrintResults(dist, min(30, num_nodes));

		utilities::CompareArrays(cpu_dist, dist, num_nodes);
	}

	if(arguments.hasOutput)
		utilities::SaveResults(arguments.output, dist, num_nodes);

	gpuErrorcheck(cudaFree(d_nodePointer));
	gpuErrorcheck(cudaFree(d_edgeList));
	gpuErrorcheck(cudaFree(d_dist));
	gpuErrorcheck(cudaFree(d_finished));
	gpuErrorcheck(cudaFree(d_finished2));
	gpuErrorcheck(cudaFree(d_label1));
	gpuErrorcheck(cudaFree(d_label2));
	gpuErrorcheck(cudaFree(d_partNodePointer));
}
