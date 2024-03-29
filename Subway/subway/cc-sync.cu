#include "../shared/subway_globals.hpp"
#include "../shared/timer.hpp"
#include "../shared/subway_argument_parsing.cuh"
#include "../shared/subway_graph.cuh"
#include "../shared/subgraph.cuh"
#include "../shared/partitioner.cuh"
#include "../shared/subgraph_generator.cuh"
#include "../shared/gpu_error_check.cuh"
#include "../shared/gpu_kernels.cuh"
#include "../shared/subway_utilities.hpp"
#include "../shared/nvmlClass.cuh"

int main(int argc, char** argv)
{
	cudaFree(0);

	SubwayArgumentParser arguments(argc, argv, false, false);
	
	// Energy structures initilization
	// Two cpu threads are used to coordinate energy consumption by chanding common flags in nvmlClass
	vector<thread> cpu_threads;
	nvmlClass nvml(arguments.deviceID, arguments.energyFile, arguments.energyStats, (string) "subway-async");

	if (arguments.energy) {
		cout << "Starting energy measurements. Timing information will be affected..." << endl;

		cpu_threads.emplace_back(std::thread(&nvmlClass::getStats, &nvml));

  		nvml.log_start();
	}

	Timer timer;
	timer.Start();
	
	SubwayGraph<OutEdge> graph(arguments.input, false);
	graph.ReadGraph();
	
	float readtime = timer.Finish();
	cout << "Graph Reading finished in " << readtime << " (ms).\n";
	
	Timer totalTimer;
	totalTimer.Start();
	if (arguments.energy) nvml.log_point();

	for(unsigned int i=0; i<graph.num_nodes; i++)
	{
		graph.value[i] = i;
		graph.label1[i] = false;
		graph.label2[i] = true;
	}

	gpuErrorcheck(cudaMemcpy(graph.d_outDegree, graph.outDegree, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(graph.d_value, graph.value, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(graph.d_label1, graph.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(graph.d_label2, graph.label2, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	
	Subgraph<OutEdge> subgraph(graph.num_nodes, graph.num_edges);
	
	SubgraphGenerator<OutEdge> subgen(graph);
	
	subgen.generate(graph, subgraph);


	Partitioner<OutEdge> partitioner;
	
	timer.Start();
	
	uint itr = 0;
	
	if (arguments.energy) nvml.log_point();
	while (subgraph.numActiveNodes>0)
	{
		itr++;
		
		partitioner.partition(subgraph, subgraph.numActiveNodes);
		// a super iteration
		for(int i=0; i<partitioner.numPartitions; i++)
		{
			cudaDeviceSynchronize();
			gpuErrorcheck(cudaMemcpy(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[i], (partitioner.partitionEdgeSize[i]) * sizeof(OutEdge), cudaMemcpyHostToDevice));
			cudaDeviceSynchronize();

			moveUpLabels<<< partitioner.partitionNodeSize[i]/512 + 1 , 512 >>>(subgraph.d_activeNodes, graph.d_label1, graph.d_label2, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);

			cc_kernel<<< partitioner.partitionNodeSize[i]/512 + 1 , 512 >>>(partitioner.partitionNodeSize[i],
													partitioner.fromNode[i],
													partitioner.fromEdge[i],
													subgraph.d_activeNodes,
													subgraph.d_activeNodesPointer,
													subgraph.d_activeEdgeList,
													graph.d_outDegree,
													graph.d_value, 
													//d_finished,
													graph.d_label1,
													graph.d_label2);

			cudaDeviceSynchronize();
			gpuErrorcheck( cudaPeekAtLastError() );	
		}
		
		subgen.generate(graph, subgraph);
			
	}	
	if (arguments.energy) nvml.log_point();
	gpuErrorcheck(cudaMemcpy(graph.value, graph.d_value, graph.num_nodes*sizeof(uint), cudaMemcpyDeviceToHost));
	if (arguments.energy) nvml.log_point();
	
	float runtime = timer.Finish();
	float total = totalTimer.Finish();
	cout << "Processing finished in " << runtime << " (ms).\n";
	cout << "Total GPU activity finished in " << total << " (ms).\n";
	
	// Stop measuring energy consumption, clean up structures
	if (arguments.energy) {
		cpu_threads.emplace_back(thread( &nvmlClass::killThread, &nvml));

		for (auto& th : cpu_threads) {
			th.join();
			th.~thread();
		}

		cpu_threads.clear();
	}

	utilities::PrintResults(graph.value, min(30, graph.num_nodes));
			
	if(arguments.hasOutput)
		utilities::SaveResults(arguments.output, graph.value, graph.num_nodes);
}

