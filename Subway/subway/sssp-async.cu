#include "../shared/globals.hpp"
#include "../shared/timer.hpp"
#include "../shared/argument_parsing.cuh"
#include "../shared/graph.cuh"
#include "../shared/subgraph.cuh"
#include "../shared/partitioner.cuh"
#include "../shared/subgraph_generator.cuh"
#include "../shared/gpu_error_check.cuh"
#include "../shared/gpu_kernels.cuh"
#include "../shared/subway_utilities.hpp"
#include "../shared/test.cuh"
#include "../shared/test.cu"
#include "../shared/nvmlClass.cuh"

int main(int argc, char** argv)
{
	/*
	Test<int> test;
	cout << test.sum(20, 30) << endl;
	*/
	
	cudaFree(0);

	ArgumentParser arguments(argc, argv, true, false);
	
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
	
	Graph<OutEdgeWeighted> graph(arguments.input, true);
	graph.ReadGraph();

	float readtime = timer.Finish();
	cout << "Graph Reading finished in " << readtime/1000 << " (s).\n";
		
	Timer totalTimer;
	totalTimer.Start();
	if (arguments.energy) nvml.log_point();

	//for(unsigned int i=0; i<100; i++)
	//	cout << graph.edgeList[i].end << " " << graph.edgeList[i].w8;
	
	for(unsigned int i=0; i<graph.num_nodes; i++)
	{
		graph.value[i] = DIST_INFINITY;
		graph.label1[i] = true;
		graph.label2[i] = false;
	}
	graph.value[arguments.sourceNode] = 0;
	//graph.label[arguments.sourceNode] = true;


	gpuErrorcheck(cudaMemcpy(graph.d_outDegree, graph.outDegree, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(graph.d_value, graph.value, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(graph.d_label1, graph.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(graph.d_label2, graph.label2, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	
	Subgraph<OutEdgeWeighted> subgraph(graph.num_nodes, graph.num_edges);
	
	SubgraphGenerator<OutEdgeWeighted> subgen(graph);
	
	subgen.generate(graph, subgraph);
	
	for(unsigned int i=0; i<graph.num_nodes; i++)
	{
		graph.label1[i] = false;
	}
	graph.label1[arguments.sourceNode] = true;
	gpuErrorcheck(cudaMemcpy(graph.d_label1, graph.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));	
	

	Partitioner<OutEdgeWeighted> partitioner;
	
	timer.Start();
	
	uint gItr = 0;
	
	bool finished;
	bool *d_finished;
	gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));
	
	if (arguments.energy) nvml.log_point();
	while (subgraph.numActiveNodes>0)
	{
		gItr++;
		
		partitioner.partition(subgraph, subgraph.numActiveNodes);
		// a super iteration
		for(int i=0; i<partitioner.numPartitions; i++)
		{
			cudaDeviceSynchronize();
			gpuErrorcheck(cudaMemcpy(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[i], (partitioner.partitionEdgeSize[i]) * sizeof(OutEdgeWeighted), cudaMemcpyHostToDevice));
			cudaDeviceSynchronize();

			//moveUpLabels<<< partitioner.partitionNodeSize[i]/512 + 1 , 512 >>>(subgraph.d_activeNodes, graph.d_label, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);
			mixLabels<<<partitioner.partitionNodeSize[i]/512 + 1 , 512>>>(subgraph.d_activeNodes, graph.d_label1, graph.d_label2, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);
			
			uint itr = 0;
			do
			{
				itr++;
				finished = true;
				gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

				sssp_async<<< partitioner.partitionNodeSize[i]/512 + 1 , 512 >>>(partitioner.partitionNodeSize[i],
													partitioner.fromNode[i],
													partitioner.fromEdge[i],
													subgraph.d_activeNodes,
													subgraph.d_activeNodesPointer,
													subgraph.d_activeEdgeList,
													graph.d_outDegree,
													graph.d_value, 
													d_finished,
													(itr%2==1) ? graph.d_label1 : graph.d_label2,
													(itr%2==1) ? graph.d_label2 : graph.d_label1);													

				cudaDeviceSynchronize();
				gpuErrorcheck( cudaPeekAtLastError() );	
				
				gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
			}while(!(finished));
			
			cout << itr << ((itr>1) ? " Inner Iterations" : " Inner Iteration") << " in Global Iteration " << gItr << ", Partition " << i  << endl;			
		}
		
		subgen.generate(graph, subgraph);
			
	}
	if (arguments.energy) nvml.log_point();
	gpuErrorcheck(cudaMemcpy(graph.value, graph.d_value, graph.num_nodes*sizeof(uint), cudaMemcpyDeviceToHost));
	if (arguments.energy) nvml.log_point();
	
	float runtime = timer.Finish();
	float total = totalTimer.Finish();
	cout << "Processing finished in " << runtime/1000 << " (s).\n";
	cout << "Total GPU activity finished in " << total/1000 << " (s).\n";
	
	// Stop measuring energy consumption, clean up structures
	if (arguments.energy) {
		cpu_threads.emplace_back(thread( &nvmlClass::killThread, &nvml));

		for (auto& th : cpu_threads) {
			th.join();
			th.~thread();
		}

		cpu_threads.clear();
	}

	utilities::PrintResults(graph.value, min(20, graph.num_nodes));
			
	if(arguments.hasOutput)
		utilities::SaveResults(arguments.output, graph.value, graph.num_nodes);
}

