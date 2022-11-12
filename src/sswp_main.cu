#include "../include/timer.hpp"
#include "../include/utilities.hpp"
#include "../include/graph.hpp"
#include "../include/globals.hpp"
#include "../include/argument_parsing.hpp"
#include "../include/gpu_error_check.cuh"
#include "../include/cuda_includes.cuh"
#include "../include/nvmlClass.cuh"
#include "../include/sswp.cuh"
#include "../include/virtual_graph.hpp"
#include <iostream>

/*int main_subway(ArgumentParser arguments) {
	cout << "Subway graph partitioning" << endl;
	exit(0);
}*/

// Tried this but kept getting segfaults
/*long long MakeGraphUM(Graph* graph, uint* nodePointer, uint* edgeList, PartPointer* partNodePointer, uint* outDegree)
{ 
	/*nodePointer = new uint[graph->num_nodes];
	edgeList = new uint[2*graph->num_edges + graph->num_nodes];*/
	/*cout << "Entered function" << endl;
	gpuErrorcheck(cudaMallocManaged(&nodePointer, graph->num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMallocManaged(&edgeList, (2*graph->num_edges + graph->num_nodes) * sizeof(unsigned int)));
	
	uint outDegreeCounter[graph->num_nodes];
	uint source;
	uint end;
	uint w8;		
	
	
	long long counter=0;
	long long numParts = 0;
	int numZero = 0;
	
	for(int i=0; i<graph->num_nodes; i++)
	{
		nodePointer[i] = counter;
		edgeList[counter] = outDegree[i];
		
		if(outDegree[i] == 0)
			numZero++;
		
		if(outDegree[i] % Part_Size == 0)
			numParts += outDegree[i] / Part_Size ;
		else
			numParts += outDegree[i] / Part_Size + 1;
		
		counter = counter + outDegree[i]*2 + 1;
	}

	//outDegreeCounter  = new uint[graph->num_nodes];
	
	for(int i=0; i<graph->num_edges; i++)
	{
		source = graph->edges[i].source;
		end = graph->edges[i].end;
		w8 = graph->weights[i];
		
		uint location = nodePointer[source]+1+2*outDegreeCounter[source];

		cout << "Nodepointer:" << nodePointer[source] << endl;
		cout << "Outdeg:" << outDegreeCounter[source] << endl; // Outdeg is larger than graph!
		cout << "Source:" << source << endl;
		cout << "Location:" << location << endl;

		edgeList[location] = end; // Seg faults here
		edgeList[location+1] = w8;

		outDegreeCounter[source]++;  
	}
	
	
	//partNodePointer = new PartPointer[numParts];

	gpuErrorcheck(cudaMallocManaged(&partNodePointer, numParts * sizeof(PartPointer)));
	
	int thisNumParts;
	long long countParts = 0;
	for(int i=0; i<graph->num_nodes; i++)
	{
		if(outDegree[i] % Part_Size == 0)
			thisNumParts = outDegree[i] / Part_Size ;
		else
			thisNumParts = outDegree[i] / Part_Size + 1;
		for(int j=0; j<thisNumParts; j++)
		{
			partNodePointer[countParts].node = i;
			partNodePointer[countParts++].part = j;
		}
	}

	return numParts;
}*/

int main_unified_memory(ArgumentParser arguments) {
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
	//long long numParts;

	gpuErrorcheck(cudaMallocManaged(&finished, sizeof(bool)));
	gpuErrorcheck(cudaMallocManaged(&finished2, sizeof(bool)));

	gpuErrorcheck(cudaMallocManaged(&nodePointer, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMallocManaged(&edgeList, (2*num_edges + num_nodes) * sizeof(unsigned int)));
	gpuErrorcheck(cudaMallocManaged(&partNodePointer, vGraph.numParts * sizeof(PartPointer)));

	//numParts = MakeGraphUM(vGraph.graph, nodePointer, edgeList, partNodePointer, vGraph.outDegree);

	// Copy from structures into unified memory versions of structures. 
	// Doubles memory size?
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
		do
		{
			itr++;
			*finished = true;
			if(itr % 2 == 1)
			{
				sswp::sync_push_dd<<< num_blocks , num_threads >>>(vGraph.numParts, 
															nodePointer,
															partNodePointer,
															edgeList, 
															dist, 
															finished,
															label1,
															label2);
				sswp::clearLabel<<< num_blocks , num_threads >>>(label1, num_nodes);
			}
			else
			{
				sswp::sync_push_dd<<< num_blocks , num_threads >>>(vGraph.numParts, 
															nodePointer, 
															partNodePointer,
															edgeList, 
															dist, 
															finished,
															label2,
															label1);
				sswp::clearLabel<<< num_blocks , num_threads >>>(label2, num_nodes);
			}

			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck( cudaDeviceSynchronize() );
		} while (!(finished));
	} else if (arguments.variant == ASYNC_PUSH_TD) {
		do
		{
			itr++;
			*finished = true;

			sswp::async_push_td<<< num_blocks , num_threads >>>(vGraph.numParts, 
														nodePointer,
														partNodePointer,
														edgeList, 
														dist, 
														finished);

			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck( cudaDeviceSynchronize() );	
			
			gpuErrorcheck(cudaMemcpy(&finished, finished, sizeof(bool), cudaMemcpyDeviceToHost));

		} while (!(finished));
	} else if (arguments.variant == SYNC_PUSH_TD) {
		do
		{
			itr++;
			if(itr % 2 == 1)
			{
				*finished = true;
				gpuErrorcheck(cudaMemcpy(finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

				sswp::sync_push_td<<< num_blocks , num_threads >>>(vGraph.numParts, 
															nodePointer,
															partNodePointer,
															edgeList, 
															dist, 
															finished,
															false);
			}
			else
			{
				*finished2 = true;
				sswp::sync_push_td<<< num_blocks , num_threads >>>(vGraph.numParts, 
															nodePointer, 
															partNodePointer,
															edgeList, 
															dist, 
															finished2,
															true);
			}

			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck( cudaDeviceSynchronize() );
		} while (!(finished) && !(finished2));
	} else if (arguments.variant == ASYNC_PUSH_DD) {
		do
		{
			itr++;
			*finished = true;

			sswp::async_push_dd<<< num_blocks , num_threads >>>(vGraph.numParts, 
														nodePointer,
														partNodePointer,
														edgeList, 
														dist, 
														finished,
														(itr%2==1) ? label1 : label2,
														(itr%2==1) ? label2 : label1);

			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck( cudaDeviceSynchronize() );
		} while (!(finished));
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
	if (arguments.debug) {
		unsigned int* cpu_dist;
		cpu_dist = new unsigned int[num_nodes];

		for(int i=0; i<num_nodes; i++)
		{
			cpu_dist[i] = DIST_INFINITY;
		}
		
		cpu_dist[arguments.sourceNode] = 0;

		sswp::seq_cpu(	graph.edges, 
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

	if(arguments.hasOutput)
		utilities::SaveResults(arguments.output, dist, num_nodes);

	gpuErrorcheck(cudaFree(dist));
	gpuErrorcheck(cudaFree(label1));
	gpuErrorcheck(cudaFree(label2));
	gpuErrorcheck(cudaFree(nodePointer));
	gpuErrorcheck(cudaFree(edgeList));
	gpuErrorcheck(cudaFree(partNodePointer));

	exit(0);
}

int main(int argc, char** argv) {

	ArgumentParser arguments(argc, argv, true, false);

	if (arguments.unifiedMem) {
		main_unified_memory(arguments);
	} else if (arguments.subway) {
		/*main_subway(arguments);*/
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

	/*if (!sswp::checkSize(graph, vGraph, arguments.deviceID)) {
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
			label1[i] = false;
			label2[i] = false;
	}
	
	dist[arguments.sourceNode] = 0;
	label1[arguments.sourceNode] = true;

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
				sswp::sync_push_dd<<< num_blocks , num_threads >>>(vGraph.numParts, 
															d_nodePointer,
															d_partNodePointer,
															d_edgeList, 
															d_dist, 
															d_finished,
															d_label1,
															d_label2);
				sswp::clearLabel<<< num_blocks , num_threads >>>(d_label1, num_nodes);
			}
			else
			{
				sswp::sync_push_dd<<< num_blocks , num_threads >>>(vGraph.numParts, 
															d_nodePointer, 
															d_partNodePointer,
															d_edgeList, 
															d_dist, 
															d_finished,
															d_label2,
															d_label1);
				sswp::clearLabel<<< num_blocks , num_threads >>>(d_label2, num_nodes);
			}

			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck( cudaDeviceSynchronize() );	
			
			gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
			

		} while (!(finished));
	} else if (arguments.variant == ASYNC_PUSH_TD) {
		do
		{
			itr++;
			finished = true;
			gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

			sswp::async_push_td<<< num_blocks , num_threads >>>(vGraph.numParts, 
														d_nodePointer,
														d_partNodePointer,
														d_edgeList, 
														d_dist, 
														d_finished);

			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck( cudaDeviceSynchronize() );	
			
			gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
			

		} while (!(finished));
	} else if (arguments.variant == SYNC_PUSH_TD) {
		do
		{
			itr++;
			if(itr % 2 == 1)
			{
				finished = true;
				gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

				sswp::sync_push_td<<< num_blocks , num_threads >>>(vGraph.numParts, 
															d_nodePointer,
															d_partNodePointer,
															d_edgeList, 
															d_dist, 
															d_finished,
															false);
			}
			else
			{
				finished2 = true;
				gpuErrorcheck(cudaMemcpy(d_finished2, &finished2, sizeof(bool), cudaMemcpyHostToDevice));
				sswp::sync_push_td<<< num_blocks , num_threads >>>(vGraph.numParts, 
															d_nodePointer, 
															d_partNodePointer,
															d_edgeList, 
															d_dist, 
															d_finished2,
															true);
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

			sswp::async_push_dd<<< num_blocks , num_threads >>>(vGraph.numParts, 
														d_nodePointer,
														d_partNodePointer,
														d_edgeList, 
														d_dist, 
														d_finished,
														(itr%2==1) ? d_label1 : d_label2,
														(itr%2==1) ? d_label2 : d_label1);

			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck( cudaDeviceSynchronize() );	
			
			gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
			

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

		sswp::seq_cpu(	graph.edges, 
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
