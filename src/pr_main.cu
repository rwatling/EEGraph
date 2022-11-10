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

/*int main_subway(ArgumentParser arguments) {
	cout << "Subway graph partitioning" << endl;
	exit(0);
}*/

/*int main_unified_memory(ArgumentParser arguments) {
	cout << "Unified memory version" << endl;
		
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
	graph.ReadGraph();

	VirtualGraph vGraph(graph);
	vGraph.MakeGraph();

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
	bool *label1;
	bool *label2;

	if (arguments.energy) nvml.log_point();

	gpuErrorcheck(cudaMallocManaged(&dist, sizeof(unsigned int) * num_nodes));
	gpuErrorcheck(cudaMallocManaged(&label1, sizeof(bool) * num_nodes));
	gpuErrorcheck(cudaMallocManaged(&label2, sizeof(bool) * num_nodes));
	
	for(int i=0; i<num_nodes; i++)
	{
			dist[i] = DIST_INFINITY;
			label1[i] = false;
			label2[i] = false;
	}
	
	dist[arguments.sourceNode] = 0;
	label1[arguments.sourceNode] = true;

	uint *nodePointer;
	uint *edgeList;
	PartPointer *partNodePointer; 
	bool *finished;
	bool *finished2;

	gpuErrorcheck(cudaMallocManaged(&nodePointer, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMallocManaged(&edgeList, (2*num_edges + num_nodes) * sizeof(unsigned int)));
	gpuErrorcheck(cudaMallocManaged(&finished, sizeof(bool)));
	gpuErrorcheck(cudaMallocManaged(&finished2, sizeof(bool)));
	gpuErrorcheck(cudaMallocManaged(&partNodePointer, vGraph.numParts * sizeof(PartPointer)));

	// Copy from structures into unified memory versions of structures. 
	// In the future, this could be done in the class defintion
	//Node pointer
	for (int i = 0; i < num_nodes; i++) {
		nodePointer[i] = vGraph.nodePointer[i];
	}

	//Edgelist
	for (int i = 0; i < (2 * num_edges + num_nodes); i++) {
		edgeList[i] = vGraph.edgeList[i];
	}
	//partNodePointer
	for (int i = 0; i < (vGraph.numParts); i++) {
		partNodePointer[i] = vGraph.partNodePointer[i];
	}

	// Tell GPU this data is mostly read
	gpuErrorcheck(cudaMemAdvise(nodePointer, num_nodes * sizeof(unsigned int), cudaMemAdviseSetReadMostly, arguments.deviceID));
	gpuErrorcheck(cudaMemAdvise(edgeList, (2*num_edges + num_nodes) * sizeof(unsigned int), cudaMemAdviseSetReadMostly, arguments.deviceID));
	gpuErrorcheck(cudaMemAdvise(partNodePointer, vGraph.numParts * sizeof(PartPointer), cudaMemAdviseSetReadMostly, arguments.deviceID));

	// Recommend that most of the data are just used on device
	gpuErrorcheck(cudaMemAdvise(nodePointer, num_nodes * sizeof(unsigned int), cudaMemAdviseSetPreferredLocation, arguments.deviceID));
	gpuErrorcheck(cudaMemAdvise(edgeList, (2*num_edges + num_nodes) * sizeof(unsigned int), cudaMemAdviseSetPreferredLocation, arguments.deviceID));
	gpuErrorcheck(cudaMemAdvise(partNodePointer, vGraph.numParts * sizeof(PartPointer), cudaMemAdviseSetPreferredLocation, arguments.deviceID));

	if (arguments.energy) nvml.log_point();

	// Algorithm control variable declarations
	Timer timer;
	int itr = 0;
	uint num_threads = 512;
	uint num_blocks = vGraph.numParts / num_threads + 1;

	timer.Start();

	if (arguments.variant == SYNC_PUSH_DD) {
	} else if (arguments.variant == ASYNC_PUSH_TD) {
	} else if (arguments.variant == SYNC_PUSH_TD) {
	} else if (arguments.variant == ASYNC_PUSH_DD) {
	}

	// Stop measuring energy consumption, clean up structures
	if (arguments.energy) {
		cpu_threads.emplace_back(thread( &nvmlClass::killThread, &nvml));

		for (auto& th : cpu_threads) {
			th.join();
			th.~thread();
		}

		cpu_threads.clear();
	}

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

	if(arguments.hasOutput)
		utilities::SaveResults(arguments.output, dist, num_nodes);

	gpuErrorcheck(cudaFree(dist));
	gpuErrorcheck(cudaFree(label1));
	gpuErrorcheck(cudaFree(label2));
	gpuErrorcheck(cudaFree(nodePointer));
	gpuErrorcheck(cudaFree(edgeList));
	gpuErrorcheck(cudaFree(partNodePointer));

	exit(0);
}*/

int main(int argc, char** argv) {

	ArgumentParser arguments(argc, argv, false, true);

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

	float *pr1, *pr2;
	pr1 = new float[num_nodes];
	pr2 = new float[num_nodes];

	float initPR = (float) 1 / num_nodes;
	
	for(int i=0; i<num_nodes; i++)
	{
		pr1[i] = 0;
		pr2[i] = initPR;
		label1[i] = false;
		label2[i] = false;
	}

	label1[arguments.sourceNode] = true;

	uint *d_nodePointer;
	uint *d_edgeList;
	PartPointer *d_partNodePointer; 
	bool *d_label1;
	bool *d_label2;
	float *d_pr1;
	float *d_pr2;

	if (arguments.energy) nvml.log_point();

	gpuErrorcheck(cudaMalloc(&d_nodePointer, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_edgeList, (2*num_edges + num_nodes) * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_pr1, num_nodes * sizeof(float)));
	gpuErrorcheck(cudaMalloc(&d_pr2, num_nodes * sizeof(float)));
	gpuErrorcheck(cudaMalloc(&d_label1, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label2, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_partNodePointer, vGraph.numParts * sizeof(PartPointer)));

	gpuErrorcheck(cudaMemcpy(d_nodePointer, vGraph.nodePointer, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_edgeList, vGraph.edgeList, (2*num_edges + num_nodes) * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_pr1, pr1, num_nodes * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_pr2, pr2, num_nodes * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_partNodePointer, vGraph.partNodePointer, vGraph.numParts * sizeof(PartPointer), cudaMemcpyHostToDevice));

	if (arguments.energy) nvml.log_point();

	// Algorithm control variable declarations
	Timer timer;
	int itr = 0;
	uint num_threads = 512;
	uint num_blocks = vGraph.numParts / num_threads + 1;
	float base = (float)0.15/num_nodes;

	timer.Start();

	if (arguments.variant == SYNC_PUSH_DD) {
	} else if (arguments.variant == ASYNC_PUSH_TD) {
	} else if (arguments.variant == SYNC_PUSH_TD) {
		do
		{
			itr++;
			if(itr % 2 == 1)
			{
				pr::sync_push_td<<< num_blocks , num_threads >>>(vGraph.numParts, 
															d_nodePointer,
															d_partNodePointer,
															d_edgeList, 
															d_pr1,
															d_pr2);
				pr::clearVal<<< num_blocks , num_threads >>>(d_pr1, d_pr2, num_nodes, base);	
			}
			else
			{
				pr::sync_push_td<<<num_blocks , num_threads >>>(vGraph.numParts, 
															d_nodePointer, 
															d_partNodePointer,
															d_edgeList,
															d_pr2,
															d_pr1);
				pr::clearVal<<< num_blocks , num_threads >>>(d_pr2, d_pr1, num_nodes, base);												
			}
		
			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck( cudaDeviceSynchronize() );
			
		} while(itr < arguments.numberOfItrs);
	} else if (arguments.variant == ASYNC_PUSH_DD) {
	}

	if(itr % 2 == 1)
	{
		gpuErrorcheck(cudaMemcpy(pr1, d_pr1, num_nodes*sizeof(float), cudaMemcpyDeviceToHost));
	}
	else
	{
		gpuErrorcheck(cudaMemcpy(pr1, d_pr2, num_nodes*sizeof(float), cudaMemcpyDeviceToHost));
	}

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

	// Naive checks
	if (arguments.debug) {
		float sum = 0;
		for (int i = 0; i < num_nodes; i++) {
			sum += pr1[i];
			if (pr1[i] < 0) {
				cout << "Error: negative node value at " << i << endl;
			}
		}

		cout << "Sum: " << sum << endl;
		
		if (num_nodes < 20) {
			utilities::PrintResults(pr1, num_nodes);
		} else {
			utilities::PrintResults(pr1, 20);
		}
	}

	if(arguments.hasOutput)
		utilities::SaveResults(arguments.output, pr1, num_nodes);

	gpuErrorcheck(cudaFree(d_nodePointer));
	gpuErrorcheck(cudaFree(d_edgeList));
	gpuErrorcheck(cudaFree(d_pr1));
	gpuErrorcheck(cudaFree(d_pr2));
	gpuErrorcheck(cudaFree(d_label1));
	gpuErrorcheck(cudaFree(d_label2));
	gpuErrorcheck(cudaFree(d_partNodePointer));
}
