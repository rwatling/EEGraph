#include "../include/timer.hpp"
#include "../include/utilities.hpp"
#include "../include/graph.hpp"
#include "../include/globals.hpp"
#include "../include/argument_parsing.hpp"
#include "../include/gpu_error_check.cuh"
#include "../include/cuda_includes.cuh"
#include "../include/nvmlClass.cuh"
#include "../include/pr.cuh"
#include "../include/virtual_graph.hpp"
#include <iostream>

int main(int argc, char** argv) {

	ArgumentParser arguments(argc, argv, true, true);

	if (arguments.unifiedMem) {
		//main_unified_memory(arguments);
	} else if (arguments.subway) {
		//main_subway(arguments);
		cout << "Subway not yet implemented" << endl;
	}

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

	/*if (!pr::checkSize(graph, vGraph, arguments.deviceID)) {
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

	bool *label1;
	bool *label2;
	label1 = new bool[num_nodes];
	label2 = new bool[num_nodes];

	float *delta, *value;
	delta = new float[num_nodes];
	value = new float[num_nodes];


	float initPR = (float) 1 / num_nodes;
	float acc = arguments.acc;
	
	for(int i=0; i<num_nodes; i++)
	{
		delta[i] = 0;
		value[i] = initPR;
		if ( i < (num_nodes / 4)) {
			label1[i] = true;
		} else { label1[i] = false; }
		label2[i] = false;
	}

	/*
	Note: Since PageRank is an iterative algorithm,
	more than one node should be considered active.
	Otherwise the algorithm will converge early by setting the finished flags
	to be done for the one active node which is does not satisfy the goal in for the
	graph as a whole in this implementation. Therefore, we naively consider
	a fraction of the graph to be active initially.
	*/
	label1[arguments.sourceNode] = true;

	uint *d_nodePointer;
	uint *d_edgeList;
	PartPointer *d_partNodePointer; 
	bool *d_label1;
	bool *d_label2;
	bool* d_finished;
	//bool* d_finished2;
	bool finished;
	//bool finished2;
	float *d_delta;
	float *d_value;

	if (arguments.energy) nvml.log_point();

	gpuErrorcheck(cudaMalloc(&d_nodePointer, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_edgeList, (2*num_edges + num_nodes) * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_delta, num_nodes * sizeof(float)));
	gpuErrorcheck(cudaMalloc(&d_value, num_nodes * sizeof(float)));
	gpuErrorcheck(cudaMalloc(&d_label1, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label2, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_partNodePointer, vGraph.numParts * sizeof(PartPointer)));
	gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));
	//gpuErrorcheck(cudaMalloc(&d_finished2, sizeof(bool)));

	gpuErrorcheck(cudaMemcpy(d_nodePointer, vGraph.nodePointer, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_edgeList, vGraph.edgeList, (2*num_edges + num_nodes) * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_delta, delta, num_nodes * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_value, value, num_nodes * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_partNodePointer, vGraph.partNodePointer, vGraph.numParts * sizeof(PartPointer), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_label1, label1, num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_label2, label2, num_nodes * sizeof(bool), cudaMemcpyHostToDevice));

	if (arguments.energy) nvml.log_point();

	// Algorithm control variable declarations
	Timer timer;
	int itr = 0;
	uint num_threads = 512;
	uint num_blocks = vGraph.numParts / num_threads + 1;
	//float base = (float)0.15/num_nodes;

	timer.Start();

	if (arguments.variant == SYNC_PUSH_DD) {
	} else if (arguments.variant == ASYNC_PUSH_TD) {
		do {
			itr++;
			finished = true;
			gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

			pr::async_push_td<<< num_blocks, num_threads >>>(vGraph.numParts, 
															d_nodePointer,
															d_partNodePointer,
															d_edgeList, 
															d_delta,
															d_value,
															d_finished,
															acc);
			cudaDeviceSynchronize();
			gpuErrorcheck( cudaPeekAtLastError() );	
			
			gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
		} while (!finished);
	} else if (arguments.variant == SYNC_PUSH_TD) {
		/*do
		{
			itr++;
			if(itr % 2 == 1)
			{
				finished = true;
				gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));
				
				pr::sync_push_td<<< num_blocks , num_threads >>>(vGraph.numParts, 
															d_nodePointer,
															d_partNodePointer,
															d_edgeList, 
															d_delta,
															d_value,
															d_finished,
															acc);
			}
			else
			{
				finished2 = true;
				gpuErrorcheck(cudaMemcpy(d_finished2, &finished2, sizeof(bool), cudaMemcpyHostToDevice));
				pr::sync_push_td<<<num_blocks , num_threads >>>(vGraph.numParts, 
																d_nodePointer, 
																d_partNodePointer,
																d_edgeList,
																d_value,
																d_delta,
																d_finished2,
																acc);										
			}
		
			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck( cudaDeviceSynchronize() );

			gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
			gpuErrorcheck(cudaMemcpy(&finished2, d_finished2, sizeof(bool), cudaMemcpyDeviceToHost));
			
		} while(!(finished) && !(finished2));

		if(itr % 2 == 1)
		{
			gpuErrorcheck(cudaMemcpy(delta, d_delta, num_nodes*sizeof(float), cudaMemcpyDeviceToHost));
		}
		else
		{
			gpuErrorcheck(cudaMemcpy(delta, d_value, num_nodes*sizeof(float), cudaMemcpyDeviceToHost));
		}*/
	} else if (arguments.variant == ASYNC_PUSH_DD) {
		do {
			itr++;
			finished = true;
			gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

			pr::async_push_dd<<< num_blocks, num_threads >>>(vGraph.numParts, 
															d_nodePointer,
															d_partNodePointer,
															d_edgeList, 
															d_delta,
															d_value,
															d_finished,
															acc,
															(itr%2==1) ? d_label1 : d_label2,
															(itr%2==1) ? d_label2 : d_label1);
			cudaDeviceSynchronize();
			gpuErrorcheck( cudaPeekAtLastError() );	
			
			gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
		} while (!finished);
	}

	if (arguments.energy) nvml.log_point();

	cout << "Number of iterations = " << itr << endl;

	float runtime = timer.Finish();
	cout << "Processing finished in " << runtime << " (ms).\n";

	gpuErrorcheck(cudaMemcpy(value, d_value, num_nodes*sizeof(float), cudaMemcpyDeviceToHost));

	// Stop measuring energy consumption, clean up structures
	if (arguments.energy) {
		cpu_threads.emplace_back(thread( &nvmlClass::killThread, &nvml));

		for (auto& th : cpu_threads) {
			th.join();
			th.~thread();
		}

		cpu_threads.clear();
	}

	// Print results
	if (arguments.debug) {		
		utilities::PrintResults(value, min(30, num_nodes));
	}

	if(arguments.hasOutput)
		utilities::SaveResults(arguments.output, delta, num_nodes);

	gpuErrorcheck(cudaFree(d_nodePointer));
	gpuErrorcheck(cudaFree(d_edgeList));
	gpuErrorcheck(cudaFree(d_delta));
	gpuErrorcheck(cudaFree(d_value));
	gpuErrorcheck(cudaFree(d_label1));
	gpuErrorcheck(cudaFree(d_label2));
	gpuErrorcheck(cudaFree(d_partNodePointer));
}
