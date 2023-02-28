#include "../include/eegraph.cuh"

Result eegraph_bfs(ArgumentParser &arguments, Graph &graph) {

	// Energy structures initilization
	// Two cpu threads are used to coordinate energy consumption by chanding common flags in nvmlClass
	vector<thread> cpu_threads;
	nvmlClass nvml(arguments.deviceID, arguments.energyFile, arguments.energyStats, to_string(arguments.variant));

	if (arguments.energy) {
		cout << "Starting energy measurements. Timing information will be affected..." << endl;

		cpu_threads.emplace_back(std::thread(&nvmlClass::getStats, &nvml));

  		nvml.log_start();
	}

	VirtualGraph vGraph(graph);
	vGraph.MakeGraph();

	uint num_nodes = graph.num_nodes;
	uint num_edges = graph.num_edges;

	if(arguments.hasDeviceID)
		gpuErrorcheck(cudaSetDevice(arguments.deviceID));

	cudaFree(0);

	unsigned int *dist;
	dist  = new unsigned int[num_nodes];

	bool *label1;
	bool *label2;
	label1 = new bool[num_nodes];
	label2 = new bool[num_nodes];
	
	Timer totalTimer;
	totalTimer.Start();
	if (arguments.energy) nvml.log_point();

	for(int i=0; i<num_nodes; i++)
	{
		dist[i] = DIST_INFINITY;
		if (arguments.variant == ASYNC_PUSH_DD)	label1[i] = true;
		else label1[i]=false;
		label2[i] = false;
	}
	label1[arguments.sourceNode] = true;
	dist[arguments.sourceNode] = 0;

	uint *d_nodePointer;
	uint *d_edgeList;
	uint *d_dist;
	PartPointer *d_partNodePointer; 
	bool *d_label1;
	bool *d_label2;
	
	bool finished;
	bool *d_finished;

	gpuErrorcheck(cudaMalloc(&d_nodePointer, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_edgeList, (2*num_edges + num_nodes) * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_dist, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label1, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label2, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_partNodePointer, vGraph.numParts * sizeof(PartPointer)));

	gpuErrorcheck(cudaMemcpy(d_nodePointer, vGraph.nodePointer, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_edgeList, vGraph.edgeList, (2*num_edges + num_nodes) * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_dist, dist, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_label1, label1, num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_label2, label2, num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_partNodePointer, vGraph.partNodePointer, vGraph.numParts * sizeof(PartPointer), cudaMemcpyHostToDevice));

	// Algorithm control variable declarations
	Timer timer;
	int itr = 0;
	int num_threads = 512;
	int num_blocks = vGraph.numParts / num_threads + 1;

	timer.Start();
	if (arguments.energy) nvml.log_point();

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
																d_label2);
				clearLabel<<< num_blocks , num_threads >>>(d_label1, num_nodes);
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
																d_label1);
				clearLabel<<< num_blocks , num_threads >>>(d_label2, num_nodes);
			}

			gpuErrorcheck( cudaDeviceSynchronize() );
			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
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
														d_finished);

			gpuErrorcheck( cudaDeviceSynchronize() );
			gpuErrorcheck( cudaPeekAtLastError() );	
			gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
		} while (!(finished));
	} else if (arguments.variant == SYNC_PUSH_TD) {
		do
		{
			itr++;
			finished = true;
			gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

			bfs::sync_push_td<<< num_blocks , num_threads >>>(vGraph.numParts, 
														d_nodePointer,
														d_partNodePointer,
														d_edgeList, 
														d_dist, 
														d_finished,
														(itr % 2 == 1) ? true : false);
			
			gpuErrorcheck( cudaDeviceSynchronize() );	
			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
		} while (!(finished));
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
														(itr%2==1) ? d_label2 : d_label1);

			gpuErrorcheck( cudaDeviceSynchronize() );	
			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
		} while (!(finished));
	}

	if (arguments.energy) nvml.log_point();
	gpuErrorcheck(cudaMemcpy(dist, d_dist, num_nodes*sizeof(unsigned int), cudaMemcpyDeviceToHost));
	if (arguments.energy) nvml.log_point();

	float runtime = timer.Finish();
	float total = totalTimer.Finish();
	cout << "Number of iterations = " << itr << endl;
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

	Result result;
	result.time = total;
	result.energy = nvml.get_energy();

	// Print out
	if (arguments.debug) {

		unsigned int* cpu_dist;
		cpu_dist = new unsigned int[num_nodes];

		for(int i=0; i<num_nodes; i++)
		{
			cpu_dist[i] = DIST_INFINITY;
		}
		
		cpu_dist[arguments.sourceNode] = 0;

		bfs::seq_cpu(vGraph, cpu_dist);

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
	gpuErrorcheck(cudaFree(d_label1));
	gpuErrorcheck(cudaFree(d_label2));
	gpuErrorcheck(cudaFree(d_partNodePointer));

	return result;
}

Result eegraph_bfs_um(ArgumentParser &arguments, UMGraph &graph) {
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

	UMVirtualGraph vGraph(graph);
	vGraph.MakeGraph();

	uint num_nodes = graph.num_nodes;
	uint num_edges = graph.num_edges;

	if(arguments.hasDeviceID)
		gpuErrorcheck(cudaSetDevice(arguments.deviceID));

	cudaFree(0);

	unsigned int *dist;
	bool *label1;
	bool *label2;

	gpuErrorcheck(cudaMallocManaged(&dist, sizeof(unsigned int) * num_nodes));
	gpuErrorcheck(cudaMallocManaged(&label1, sizeof(bool) * num_nodes));
	gpuErrorcheck(cudaMallocManaged(&label2, sizeof(bool) * num_nodes));

	Timer totalTimer;
	totalTimer.Start();
	if (arguments.energy) nvml.log_point();
	
	for(int i=0; i<num_nodes; i++)
	{
		dist[i] = DIST_INFINITY;
		if (arguments.variant == ASYNC_PUSH_DD)	label1[i] = true;
		else label1[i]=false;
		label2[i] = false;
	}
	label1[arguments.sourceNode] = true;
	dist[arguments.sourceNode] = 0;

	bool *finished;

	gpuErrorcheck(cudaMallocManaged(&finished, sizeof(bool)));

	// Tell GPU this data is mostly read
	gpuErrorcheck(cudaMemAdvise(vGraph.nodePointer, num_nodes * sizeof(unsigned int), cudaMemAdviseSetReadMostly, arguments.deviceID));
	gpuErrorcheck(cudaMemAdvise(vGraph.edgeList, (2*num_edges + num_nodes) * sizeof(unsigned int), cudaMemAdviseSetReadMostly, arguments.deviceID));
	gpuErrorcheck(cudaMemAdvise(vGraph.partNodePointer, vGraph.numParts * sizeof(PartPointer), cudaMemAdviseSetReadMostly, arguments.deviceID));

	// Algorithm control variable declarations
	Timer timer;
	int itr = 0;
	int num_threads = 512;
	int num_blocks = vGraph.numParts / num_threads + 1;

	timer.Start();
	if (arguments.energy) nvml.log_point();

	if (arguments.variant == SYNC_PUSH_DD) {
		do
		{
			itr++;
			*finished = true;

			if(itr % 2 == 1)
			{
				bfs::sync_push_dd<<< num_blocks , num_threads >>>(vGraph.numParts, 
																vGraph.nodePointer,
																vGraph.partNodePointer,
																vGraph.edgeList, 
																dist, 
																finished,
																label1,
																label2);
				clearLabel<<< num_blocks , num_threads >>>(label1, num_nodes);
			}
			else
			{
				bfs::sync_push_dd<<< num_blocks , num_threads >>>(vGraph.numParts, 
																vGraph.nodePointer, 
																vGraph.partNodePointer,
																vGraph.edgeList, 
																dist, 
																finished,
																label2,
																label1);
				clearLabel<<< num_blocks , num_threads >>>(label2, num_nodes);
			}

			gpuErrorcheck( cudaDeviceSynchronize() );
			gpuErrorcheck( cudaPeekAtLastError() );
		} while (!(*finished));
	} else if (arguments.variant == ASYNC_PUSH_TD) {
		do
		{
			itr++;
			*finished = true;

			bfs::async_push_td<<< num_blocks , num_threads >>>(vGraph.numParts, 
														vGraph.nodePointer,
														vGraph.partNodePointer,
														vGraph.edgeList, 
														dist, 
														finished);

			gpuErrorcheck( cudaDeviceSynchronize() );
			gpuErrorcheck( cudaPeekAtLastError() );
		} while (!(*finished));
	} else if (arguments.variant == SYNC_PUSH_TD) {		
		do
		{
			itr++;
			*finished = true;

			bfs::sync_push_td<<< num_blocks , num_threads >>>(vGraph.numParts, 
														vGraph.nodePointer,
														vGraph.partNodePointer,
														vGraph.edgeList, 
														dist, 
														finished,
														(itr % 2 == 1) ? true : false);

			gpuErrorcheck( cudaDeviceSynchronize() );	
			gpuErrorcheck( cudaPeekAtLastError() );
		} while (!(*finished));
	} else if (arguments.variant == ASYNC_PUSH_DD) {
		do
		{
			itr++;
			*finished = true;

			bfs::async_push_dd<<< num_blocks , num_threads >>>(vGraph.numParts, 
														vGraph.nodePointer,
														vGraph.partNodePointer,
														vGraph.edgeList, 
														dist, 
														finished,
														(itr%2==1) ? label1 : label2,
														(itr%2==1) ? label2 : label1);
			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck( cudaDeviceSynchronize() );
		} while (!(*finished));
	}

	if (arguments.energy) nvml.log_point();

	float runtime = timer.Finish();
	float total = totalTimer.Finish();
	cout << "Number of iterations = " << itr << endl;
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

	Result result;
	result.time = total;
	result.energy = nvml.get_energy();

	// Run sequential cpu version and print out useful information
	if (arguments.debug) {
		unsigned int* cpu_dist;
		cpu_dist = new unsigned int[num_nodes];

		for(int i=0; i<num_nodes; i++)
		{
			cpu_dist[i] = DIST_INFINITY;
		}
		
		cpu_dist[arguments.sourceNode] = 0;

		bfs::seq_cpu(vGraph, cpu_dist);

		utilities::PrintResults(cpu_dist, min(30, num_nodes));
		utilities::PrintResults(dist, min(30, num_nodes));

		utilities::CompareArrays(cpu_dist, dist, num_nodes);
	}

	if(arguments.hasOutput)
		utilities::SaveResults(arguments.output, dist, num_nodes);

	gpuErrorcheck(cudaFree(dist));
	gpuErrorcheck(cudaFree(label1));
	gpuErrorcheck(cudaFree(label2));
	gpuErrorcheck(cudaFree(finished));
	gpuErrorcheck(cudaFree(vGraph.nodePointer));
	gpuErrorcheck(cudaFree(vGraph.edgeList));
	gpuErrorcheck(cudaFree(vGraph.partNodePointer));

	return result;
}

Result eegraph_cc(ArgumentParser &arguments, Graph &graph) {
	// Energy structures initilization
	// Two cpu threads are used to coordinate energy consumption by chanding common flags in nvmlClass
	vector<thread> cpu_threads;
	nvmlClass nvml(arguments.deviceID, arguments.energyFile, arguments.energyStats, to_string(arguments.variant));

	if (arguments.energy) {
		cout << "Starting energy measurements. Timing information will be affected..." << endl;

		cpu_threads.emplace_back(std::thread(&nvmlClass::getStats, &nvml));

  		nvml.log_start();
	}

	VirtualGraph vGraph(graph);
	vGraph.MakeGraph();

	uint num_nodes = graph.num_nodes;
	uint num_edges = graph.num_edges;

	if(arguments.hasDeviceID)
		gpuErrorcheck(cudaSetDevice(arguments.deviceID));

	cudaFree(0);

	unsigned int *dist;
	dist  = new unsigned int[num_nodes];

	bool *label1;
	bool *label2;
	label1 = new bool[num_nodes];
	label2 = new bool[num_nodes];
	
	Timer totalTimer;
	totalTimer.Start();
	if (arguments.energy) nvml.log_point();

	for(int i=0; i<num_nodes; i++)
	{
		dist[i] = i;

		if (arguments.variant == ASYNC_PUSH_DD)	label1[i] = true;
		else label1[i]=false;
		label2[i] = false;
	}

	label1[arguments.sourceNode] = true;

	uint *d_nodePointer;
	uint *d_edgeList;
	uint *d_dist;
	PartPointer *d_partNodePointer; 
	bool *d_label1;
	bool *d_label2;
	
	bool finished;
	bool *d_finished;

	gpuErrorcheck(cudaMalloc(&d_nodePointer, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_edgeList, (2*num_edges + num_nodes) * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_dist, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label1, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label2, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_partNodePointer, vGraph.numParts * sizeof(PartPointer)));

	gpuErrorcheck(cudaMemcpy(d_nodePointer, vGraph.nodePointer, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_edgeList, vGraph.edgeList, (2*num_edges + num_nodes) * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_dist, dist, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_label1, label1, num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_label2, label2, num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_partNodePointer, vGraph.partNodePointer, vGraph.numParts * sizeof(PartPointer), cudaMemcpyHostToDevice));

	// Algorithm control variable declarations
	Timer timer;
	int itr = 0;
	int num_threads = 512;
	int num_blocks = vGraph.numParts / num_threads + 1;

	timer.Start();
	if (arguments.energy) nvml.log_point();

	if (arguments.variant == SYNC_PUSH_DD) {
		do
		{
			itr++;
			finished = true;
			gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));
			if(itr % 2 == 1)
			{
				cc::sync_push_dd<<< num_blocks , num_threads >>>(vGraph.numParts, 
															d_nodePointer,
															d_partNodePointer,
															d_edgeList, 
															d_dist, 
															d_finished,
															d_label1,
															d_label2);
				clearLabel<<< num_blocks , num_threads >>>(d_label1, num_nodes);
			}
			else
			{
				cc::sync_push_dd<<< num_blocks , num_threads >>>(vGraph.numParts, 
															d_nodePointer, 
															d_partNodePointer,
															d_edgeList, 
															d_dist, 
															d_finished,
															d_label2,
															d_label1);
				clearLabel<<< num_blocks , num_threads >>>(d_label2, num_nodes);
			}

			gpuErrorcheck( cudaDeviceSynchronize() );
			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
		} while (!(finished));
	} else if (arguments.variant == ASYNC_PUSH_TD) {
		do
		{
			itr++;
			finished = true;
			gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

			cc::async_push_td<<< num_blocks , num_threads >>>(vGraph.numParts, 
																d_nodePointer,
																d_partNodePointer,
																d_edgeList, 
																d_dist, 
																d_finished);

			gpuErrorcheck( cudaDeviceSynchronize() );
			gpuErrorcheck( cudaPeekAtLastError() );	
			gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
		} while (!(finished));
	} else if (arguments.variant == SYNC_PUSH_TD) {
		do
		{
			itr++;
			finished = true;
			gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

			cc::sync_push_td<<< num_blocks , num_threads >>>(vGraph.numParts, 
														d_nodePointer,
														d_partNodePointer,
														d_edgeList, 
														d_dist, 
														d_finished,
														(itr % 2 == 1) ? true : false);
			
			gpuErrorcheck( cudaDeviceSynchronize() );	
			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
		} while (!(finished));
	} else if (arguments.variant == ASYNC_PUSH_DD) {
		do
		{
			itr++;
			finished = true;
			gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

			cc::async_push_dd<<< num_blocks , num_threads >>>(vGraph.numParts, 
														d_nodePointer,
														d_partNodePointer,
														d_edgeList, 
														d_dist, 
														d_finished,
														(itr%2==1) ? d_label1 : d_label2,
														(itr%2==1) ? d_label2 : d_label1);

			gpuErrorcheck( cudaDeviceSynchronize() );
			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
		} while (!(finished));
	}

	if (arguments.energy) nvml.log_point();
	gpuErrorcheck(cudaMemcpy(dist, d_dist, num_nodes*sizeof(unsigned int), cudaMemcpyDeviceToHost));
	if (arguments.energy) nvml.log_point();

	float runtime = timer.Finish();
	float total = totalTimer.Finish();
	cout << "Number of iterations = " << itr << endl;
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

	Result result;
	result.time = total;
	result.energy = nvml.get_energy();

	// Run sequential cpu version and print out useful information
	if (arguments.debug) {
		unsigned int* cpu_dist;
		cpu_dist = new unsigned int[num_nodes];

		for(int i=0; i<num_nodes; i++)
		{
			cpu_dist[i] = i;
		}

		cc::seq_cpu(vGraph, cpu_dist);

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
	gpuErrorcheck(cudaFree(d_label1));
	gpuErrorcheck(cudaFree(d_label2));
	gpuErrorcheck(cudaFree(d_partNodePointer));

	return result;
}

Result eegraph_cc_um(ArgumentParser &arguments, UMGraph &graph) {
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

	UMVirtualGraph vGraph(graph);
	vGraph.MakeGraph();

	uint num_nodes = graph.num_nodes;
	uint num_edges = graph.num_edges;

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

	Timer totalTimer;
	totalTimer.Start();
	if (arguments.energy) nvml.log_point();

	for(int i=0; i<num_nodes; i++)
	{
		dist[i] = i;
		if (arguments.variant == ASYNC_PUSH_DD)	label1[i] = true;
		else label1[i]=false;
		label2[i] = false;
	}
	
	label1[arguments.sourceNode] = true;
	bool *finished;

	gpuErrorcheck(cudaMallocManaged(&finished, sizeof(bool)));

	// Tell GPU this data is mostly read
	gpuErrorcheck(cudaMemAdvise(vGraph.nodePointer, num_nodes * sizeof(unsigned int), cudaMemAdviseSetReadMostly, arguments.deviceID));
	gpuErrorcheck(cudaMemAdvise(vGraph.edgeList, (2*num_edges + num_nodes) * sizeof(unsigned int), cudaMemAdviseSetReadMostly, arguments.deviceID));
	gpuErrorcheck(cudaMemAdvise(vGraph.partNodePointer, vGraph.numParts * sizeof(PartPointer), cudaMemAdviseSetReadMostly, arguments.deviceID));

	// Algorithm control variable declarations
	Timer timer;
	int itr = 0;
	int num_threads = 512;
	int num_blocks = vGraph.numParts / num_threads + 1;

	timer.Start();
	if (arguments.energy) nvml.log_point();

	if (arguments.variant == SYNC_PUSH_DD) {
		do
		{
			itr++;
			*finished = true;

			if(itr % 2 == 1)
			{
				cc::sync_push_dd<<< num_blocks , num_threads >>>(vGraph.numParts, 
															vGraph.nodePointer,
															vGraph.partNodePointer,
															vGraph.edgeList, 
															dist, 
															finished,
															label1,
															label2);
				clearLabel<<< num_blocks , num_threads >>>(label1, num_nodes);
			}
			else
			{
				cc::sync_push_dd<<< num_blocks , num_threads >>>(vGraph.numParts, 
															vGraph.nodePointer, 
															vGraph.partNodePointer,
															vGraph.edgeList, 
															dist, 
															finished,
															label2,
															label1);
				clearLabel<<< num_blocks , num_threads >>>(label2, num_nodes);
			}

			gpuErrorcheck( cudaDeviceSynchronize() );
			gpuErrorcheck( cudaPeekAtLastError() );
		} while (!(*finished));
	} else if (arguments.variant == ASYNC_PUSH_TD) {
		do
		{
			itr++;
			*finished = true;

			cc::async_push_td<<< num_blocks , num_threads >>>(vGraph.numParts, 
														vGraph.nodePointer,
														vGraph.partNodePointer,
														vGraph.edgeList, 
														dist, 
														finished);

			
			gpuErrorcheck( cudaDeviceSynchronize() );
			gpuErrorcheck( cudaPeekAtLastError() );
		} while (!(*finished));
	} else if (arguments.variant == SYNC_PUSH_TD) {
		do
		{
			itr++;

			*finished = true;

			cc::sync_push_td<<< num_blocks , num_threads >>>(vGraph.numParts, 
														vGraph.nodePointer,
														vGraph.partNodePointer,
														vGraph.edgeList, 
														dist, 
														finished,
														(itr % 2 == 1) ? true : false);
			
			gpuErrorcheck( cudaDeviceSynchronize() );
			gpuErrorcheck( cudaPeekAtLastError() );
		} while (!(*finished));
	} else if (arguments.variant == ASYNC_PUSH_DD) {
		do
		{
			itr++;
			*finished = true;

			cc::async_push_dd<<< num_blocks , num_threads >>>(vGraph.numParts, 
														vGraph.nodePointer,
														vGraph.partNodePointer,
														vGraph.edgeList, 
														dist, 
														finished,
														(itr%2==1) ? label1 : label2,
														(itr%2==1) ? label2 : label1);
			gpuErrorcheck( cudaDeviceSynchronize() );
			gpuErrorcheck( cudaPeekAtLastError() );
		} while (!(*finished));
	}

	if (arguments.energy) nvml.log_point();

	float runtime = timer.Finish();
	float total = totalTimer.Finish();
	cout << "Number of iterations = " << itr << endl;
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

	Result result;
	result.time = total;
	result.energy = nvml.get_energy();

	// Run sequential cpu version and print out useful information
	if (arguments.debug) {
		unsigned int* cpu_dist;
		cpu_dist = new unsigned int[num_nodes];

		for(int i=0; i<num_nodes; i++)
		{
			cpu_dist[i] = i;
		}
	

		cc::seq_cpu(vGraph, cpu_dist);

		if (num_nodes < 30) {
			utilities::PrintResults(cpu_dist, num_nodes);
			utilities::PrintResults(dist, num_nodes);
		} else {
			utilities::PrintResults(cpu_dist, 30);
			utilities::PrintResults(dist, 30);
		}

		utilities::CompareArrays(cpu_dist, dist, num_nodes);
	}

	if(arguments.hasOutput)
		utilities::SaveResults(arguments.output, dist, num_nodes);

	gpuErrorcheck(cudaFree(dist));
	gpuErrorcheck(cudaFree(label1));
	gpuErrorcheck(cudaFree(label2));
	gpuErrorcheck(cudaFree(finished));
	gpuErrorcheck(cudaFree(vGraph.nodePointer));
	gpuErrorcheck(cudaFree(vGraph.edgeList));
	gpuErrorcheck(cudaFree(vGraph.partNodePointer));

	return result;
}

Result eegraph_pr(ArgumentParser &arguments, Graph &graph) {
	// Energy structures initilization
	// Two cpu threads are used to coordinate energy consumption by chanding common flags in nvmlClass
	vector<thread> cpu_threads;
	nvmlClass nvml(arguments.deviceID, arguments.energyFile, arguments.energyStats, to_string(arguments.variant));

	if (arguments.energy) {
		cout << "Starting energy measurements. Timing information will be affected..." << endl;

		cpu_threads.emplace_back(std::thread(&nvmlClass::getStats, &nvml));

  		nvml.log_start();
	}

	VirtualGraph vGraph(graph);
	vGraph.MakeGraph();

	uint num_nodes = graph.num_nodes;
	uint num_edges = graph.num_edges;

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


	float initPR = 0.15;
	float acc = arguments.acc;
	
	//cout << "Initialized value: " << initPR << endl;
	//cout << "Accuracy: " << acc << endl;

	Timer totalTimer;
	totalTimer.Start();
	if (arguments.energy) nvml.log_point();

	for(int i=0; i<num_nodes; i++)
	{
		delta[i] = 0;
		value[i] = initPR;
		label1[i] = true;
		label2[i] = false;
	}

	uint *d_nodePointer;
	uint *d_edgeList;
	PartPointer *d_partNodePointer; 
	bool *d_label1;
	bool *d_label2;
	bool *d_finished;
	bool finished;
	float *d_delta;
	float *d_value;

	gpuErrorcheck(cudaMalloc(&d_nodePointer, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_edgeList, (2*num_edges + num_nodes) * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_delta, num_nodes * sizeof(float)));
	gpuErrorcheck(cudaMalloc(&d_value, num_nodes * sizeof(float)));
	gpuErrorcheck(cudaMalloc(&d_label1, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label2, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_partNodePointer, vGraph.numParts * sizeof(PartPointer)));
	gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));

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
	int num_threads = 512;
	int num_blocks = vGraph.numParts / num_threads + 1;

	timer.Start();

	if (arguments.variant == SYNC_PUSH_DD) {
		do
		{
			itr++;
			finished = true;
			gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));
			if(itr % 2 == 1)
			{
				pr::sync_push_dd<<< num_blocks , num_threads >>>(vGraph.numParts, 
																d_nodePointer,
																d_partNodePointer,
																d_edgeList, 
																d_delta,
																d_value,
																d_finished,
																acc,
																d_label1,
																d_label2);
				clearLabel<<< num_blocks , num_threads >>>(d_label1, num_nodes);
			}
			else
			{
				pr::sync_push_dd<<< num_blocks , num_threads >>>(vGraph.numParts, 
															d_nodePointer,
															d_partNodePointer,
															d_edgeList, 
															d_delta,
															d_value,
															d_finished,
															acc,
															d_label2,
															d_label1);
				clearLabel<<< num_blocks , num_threads >>>(d_label2, num_nodes);
			}

			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck( cudaDeviceSynchronize() );	
			gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
		} while (!(finished));
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
			gpuErrorcheck( cudaDeviceSynchronize() );
			gpuErrorcheck( cudaPeekAtLastError() );	
			gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
		} while (!finished);
	} else if (arguments.variant == SYNC_PUSH_TD) {
		do
		{
			itr++;
			finished = true;
			gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

				pr::sync_push_td<<< num_blocks, num_threads >>>(vGraph.numParts, 
															d_nodePointer,
															d_partNodePointer,
															d_edgeList, 
															d_delta,
															d_value,
															d_finished,
															acc,
															(itr % 2 == 1) ? true : false);

			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck( cudaDeviceSynchronize() );
			gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
		} while (!(finished));
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
			gpuErrorcheck( cudaDeviceSynchronize() );
			gpuErrorcheck( cudaPeekAtLastError() );	
			gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
		} while (!finished);
	}

	if (arguments.energy) nvml.log_point();
	gpuErrorcheck(cudaMemcpy(value, d_value, num_nodes*sizeof(float), cudaMemcpyDeviceToHost));
	if (arguments.energy) nvml.log_point();

	float runtime = timer.Finish();
	float total = totalTimer.Finish();
	cout << "Number of iterations = " << itr << endl;
	cout << "Processing finished in " << runtime << " (ms).\n";
	cout << "Total GPU activity finished in " << total << " (ms).\n";

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

	Result result;
	result.time = total;
	result.energy = nvml.get_energy();

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
	gpuErrorcheck(cudaFree(d_finished));

	return result;
}

Result eegraph_pr_um(ArgumentParser &arguments, UMGraph &graph) {
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

	UMVirtualGraph vGraph(graph);
	vGraph.MakeGraph();

	uint num_nodes = graph.num_nodes;
	uint num_edges = graph.num_edges;

	if(arguments.hasDeviceID)
		gpuErrorcheck(cudaSetDevice(arguments.deviceID));

	cudaFree(0);
	bool *label1;
	bool *label2;
	float *delta, *value;

	gpuErrorcheck(cudaMallocManaged(&delta, sizeof(float) * num_nodes));
	gpuErrorcheck(cudaMallocManaged(&value, sizeof(float) * num_nodes));
	gpuErrorcheck(cudaMallocManaged(&label1, sizeof(bool) * num_nodes));
	gpuErrorcheck(cudaMallocManaged(&label2, sizeof(bool) * num_nodes));


	float initPR = 0.15;
	float acc = arguments.acc;
	
	//cout << "Initialized value: " << initPR << endl;
	//cout << "Accuracy: " << acc << endl;

	Timer totalTimer;
	totalTimer.Start();
	if (arguments.energy) nvml.log_point();

	for(int i=0; i<num_nodes; i++)
	{
		delta[i] = 0;
		value[i] = initPR;
		label1[i] = true; //Major difference
		label2[i] = false;
	}

	bool *finished;

	gpuErrorcheck(cudaMallocManaged(&finished, sizeof(bool)));

	// Tell GPU this data is mostly read
	gpuErrorcheck(cudaMemAdvise(vGraph.nodePointer, num_nodes * sizeof(unsigned int), cudaMemAdviseSetReadMostly, arguments.deviceID));
	gpuErrorcheck(cudaMemAdvise(vGraph.edgeList, (2*num_edges + num_nodes) * sizeof(unsigned int), cudaMemAdviseSetReadMostly, arguments.deviceID));
	gpuErrorcheck(cudaMemAdvise(vGraph.partNodePointer, vGraph.numParts * sizeof(PartPointer), cudaMemAdviseSetReadMostly, arguments.deviceID));

	// Algorithm control variable declarations
	Timer timer;
	int itr = 0;
	int num_threads = 512;
	int num_blocks = vGraph.numParts / num_threads + 1;

	timer.Start();
	if (arguments.energy) nvml.log_point();

	if (arguments.variant == SYNC_PUSH_DD) {
		do
		{
			itr++;
			*finished = true;

			if(itr % 2 == 1)
			{
				pr::sync_push_dd<<< num_blocks , num_threads >>>(vGraph.numParts, 
															vGraph.nodePointer,
															vGraph.partNodePointer,
															vGraph.edgeList, 
															delta,
															value,
															finished,
															acc,
															label1,
															label2);
				clearLabel<<< num_blocks , num_threads >>>(label1, num_nodes);
			}
			else
			{
				pr::sync_push_dd<<< num_blocks , num_threads >>>(vGraph.numParts, 
															vGraph.nodePointer,
															vGraph.partNodePointer,
															vGraph.edgeList, 
															delta,
															value,
															finished,
															acc,
															label2,
															label1);
				clearLabel<<< num_blocks , num_threads >>>(label2, num_nodes);
			}

			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck( cudaDeviceSynchronize() );
		} while (!(*finished));
	} else if (arguments.variant == ASYNC_PUSH_TD) {
		do
		{
			itr++;
			*finished = true;

			pr::async_push_td<<< num_blocks , num_threads >>>(vGraph.numParts, 
															vGraph.nodePointer,
															vGraph.partNodePointer,
															vGraph.edgeList, 
															delta,
															value,
															finished,
															acc);

			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck( cudaDeviceSynchronize() );
		} while (!(*finished));
	} else if (arguments.variant == SYNC_PUSH_TD) {
		do
		{
			itr++;
			*finished = true;

				pr::sync_push_td<<< num_blocks, num_threads >>>(vGraph.numParts, 
															vGraph.nodePointer,
															vGraph.partNodePointer,
															vGraph.edgeList, 
															delta,
															value,
															finished,
															acc,
															(itr % 2 == 1) ? true : false);

			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck( cudaDeviceSynchronize() );
		} while (!(*finished));
	} else if (arguments.variant == ASYNC_PUSH_DD) {
		do
		{
			itr++;
			*finished = true;

			pr::async_push_dd<<< num_blocks , num_threads >>>(vGraph.numParts, 
														vGraph.nodePointer,
														vGraph.partNodePointer,
														vGraph.edgeList, 
														delta,
														value, 
														finished,
														acc,
														(itr%2==1) ? label1 : label2,
														(itr%2==1) ? label2 : label1);

			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck( cudaDeviceSynchronize() );
		} while (!(*finished));
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

	float runtime = timer.Finish();
	float total = totalTimer.Finish();
	cout << "Number of iterations = " << itr << endl;
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

	Result result;
	result.time = total;
	result.energy = nvml.get_energy();

	// Run sequential cpu version and print out useful information
	if (arguments.debug) {		
		utilities::PrintResults(value, min(30, num_nodes));
	}

	if(arguments.hasOutput)
		utilities::SaveResults(arguments.output, delta, num_nodes);

	gpuErrorcheck(cudaFree(delta));
	gpuErrorcheck(cudaFree(value));
	gpuErrorcheck(cudaFree(label1));
	gpuErrorcheck(cudaFree(label2));
	gpuErrorcheck(cudaFree(vGraph.nodePointer));
	gpuErrorcheck(cudaFree(vGraph.edgeList));
	gpuErrorcheck(cudaFree(vGraph.partNodePointer));
	gpuErrorcheck(cudaFree(finished));

	return result;
}

Result eegraph_sswp(ArgumentParser &arguments, Graph &graph) {
	// Energy structures initilization
	// Two cpu threads are used to coordinate energy consumption by chanding common flags in nvmlClass
	vector<thread> cpu_threads;
	nvmlClass nvml(arguments.deviceID, arguments.energyFile, arguments.energyStats, to_string(arguments.variant));

	if (arguments.energy) {
		cout << "Starting energy measurements. Timing information will be affected..." << endl;

		cpu_threads.emplace_back(std::thread(&nvmlClass::getStats, &nvml));

  		nvml.log_start();
	}

	VirtualGraph vGraph(graph);
	vGraph.MakeGraph();

	uint num_nodes = graph.num_nodes;
	uint num_edges = graph.num_edges;

	if(arguments.hasDeviceID)
		gpuErrorcheck(cudaSetDevice(arguments.deviceID));

	cudaFree(0);

	unsigned int *dist;
	dist  = new unsigned int[num_nodes];

	bool *label1;
	bool *label2;
	label1 = new bool[num_nodes];
	label2 = new bool[num_nodes];
	
	Timer totalTimer;
	totalTimer.Start();
	if (arguments.energy) nvml.log_point();

	for(int i=0; i<num_nodes; i++)
	{
		dist[i] = 0;

		if (arguments.variant == ASYNC_PUSH_DD)	label1[i] = true;
		else label1[i]=false;

		label2[i] = false;
	}
	
	dist[arguments.sourceNode] = DIST_INFINITY;
	label1[arguments.sourceNode] = true;

	uint *d_nodePointer;
	uint *d_edgeList;
	uint *d_dist;
	PartPointer *d_partNodePointer; 
	bool *d_label1;
	bool *d_label2;
	
	bool finished;
	bool *d_finished;

	gpuErrorcheck(cudaMalloc(&d_nodePointer, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_edgeList, (2*num_edges + num_nodes) * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_dist, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label1, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label2, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_partNodePointer, vGraph.numParts * sizeof(PartPointer)));

	gpuErrorcheck(cudaMemcpy(d_nodePointer, vGraph.nodePointer, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_edgeList, vGraph.edgeList, (2*num_edges + num_nodes) * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_dist, dist, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_label1, label1, num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_label2, label2, num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_partNodePointer, vGraph.partNodePointer, vGraph.numParts * sizeof(PartPointer), cudaMemcpyHostToDevice));

	// Algorithm control variable declarations
	Timer timer;
	int itr = 0;
	int num_threads = 512;
	int num_blocks = vGraph.numParts / num_threads + 1;

	timer.Start();
	if (arguments.energy) nvml.log_point();

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
				clearLabel<<< num_blocks , num_threads >>>(d_label1, num_nodes);
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
				clearLabel<<< num_blocks , num_threads >>>(d_label2, num_nodes);
			}

			gpuErrorcheck( cudaDeviceSynchronize() );
			gpuErrorcheck( cudaPeekAtLastError() );
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

			gpuErrorcheck( cudaDeviceSynchronize() );
			gpuErrorcheck( cudaPeekAtLastError() );	
			gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
		} while (!(finished));
	} else if (arguments.variant == SYNC_PUSH_TD) {
		do
		{
			itr++;
			finished = true;
			gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

			sswp::sync_push_td<<< num_blocks , num_threads >>>(vGraph.numParts, 
														d_nodePointer,
														d_partNodePointer,
														d_edgeList, 
														d_dist, 
														d_finished,
														(itr % 2 == 1) ? true : false);
			
			gpuErrorcheck( cudaDeviceSynchronize() );	
			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
		} while (!(finished));
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

			gpuErrorcheck( cudaDeviceSynchronize() );
			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
		} while (!(finished));
	}

	if (arguments.energy) nvml.log_point();
	gpuErrorcheck(cudaMemcpy(dist, d_dist, num_nodes*sizeof(unsigned int), cudaMemcpyDeviceToHost));
	if (arguments.energy) nvml.log_point();

	float runtime = timer.Finish();
	float total = totalTimer.Finish();
	cout << "Number of iterations = " << itr << endl;
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

	Result result;
	result.time = total;
	result.energy = nvml.get_energy();

	// Run sequential cpu version and print out useful information
	if (arguments.debug) {
		unsigned int* cpu_dist;
		cpu_dist = new unsigned int[num_nodes];

		for(int i=0; i<num_nodes; i++)
		{
			cpu_dist[i] = 0;
		}
		
		cpu_dist[arguments.sourceNode] = DIST_INFINITY;

		//sswp::seq_cpu(graph.edges, graph.weights, num_edges, cpu_dist);
		sswp::seq_cpu(vGraph, cpu_dist);

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
	gpuErrorcheck(cudaFree(d_label1));
	gpuErrorcheck(cudaFree(d_label2));
	gpuErrorcheck(cudaFree(d_partNodePointer));

	return result;
}

Result eegraph_sswp_um(ArgumentParser &arguments, UMGraph &graph) {
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

	UMVirtualGraph vGraph(graph);
	vGraph.MakeGraph();

	uint num_nodes = graph.num_nodes;
	uint num_edges = graph.num_edges;

	if(arguments.hasDeviceID)
		gpuErrorcheck(cudaSetDevice(arguments.deviceID));

	cudaFree(0);

	unsigned int *dist;
	bool *label1;
	bool *label2;

	gpuErrorcheck(cudaMallocManaged(&dist, sizeof(unsigned int) * num_nodes));
	gpuErrorcheck(cudaMallocManaged(&label1, sizeof(bool) * num_nodes));
	gpuErrorcheck(cudaMallocManaged(&label2, sizeof(bool) * num_nodes));
	
	Timer totalTimer;
	totalTimer.Start();
	if (arguments.energy) nvml.log_point();

	for(int i=0; i<num_nodes; i++)
	{
		dist[i] = 0;

		if (arguments.variant == ASYNC_PUSH_DD)	label1[i] = true;
		else label1[i]=false;

		label2[i] = false;
	}
	
	dist[arguments.sourceNode] = DIST_INFINITY;
	label1[arguments.sourceNode] = true;

	bool *finished;

	gpuErrorcheck(cudaMallocManaged(&finished, sizeof(bool)));

	// Tell GPU this data is mostly read
	gpuErrorcheck(cudaMemAdvise(vGraph.nodePointer, num_nodes * sizeof(unsigned int), cudaMemAdviseSetReadMostly, arguments.deviceID));
	gpuErrorcheck(cudaMemAdvise(vGraph.edgeList, (2*num_edges + num_nodes) * sizeof(unsigned int), cudaMemAdviseSetReadMostly, arguments.deviceID));
	gpuErrorcheck(cudaMemAdvise(vGraph.partNodePointer, vGraph.numParts * sizeof(PartPointer), cudaMemAdviseSetReadMostly, arguments.deviceID));

	// Algorithm control variable declarations
	Timer timer;
	int itr = 0;
	int num_threads = 512;
	int num_blocks = vGraph.numParts / num_threads + 1;

	timer.Start();
	if (arguments.energy) nvml.log_point();

	if (arguments.variant == SYNC_PUSH_DD) {
		do
		{
			itr++;
			*finished = true;

			if(itr % 2 == 1)
			{
				sswp::sync_push_dd<<< num_blocks , num_threads >>>(vGraph.numParts, 
															vGraph.nodePointer,
															vGraph.partNodePointer,
															vGraph.edgeList, 
															dist, 
															finished,
															label1,
															label2);
				clearLabel<<< num_blocks , num_threads >>>(label1, num_nodes);
			}
			else
			{
				sswp::sync_push_dd<<< num_blocks , num_threads >>>(vGraph.numParts, 
															vGraph.nodePointer, 
															vGraph.partNodePointer,
															vGraph.edgeList, 
															dist, 
															finished,
															label2,
															label1);
				clearLabel<<< num_blocks , num_threads >>>(label2, num_nodes);
			}

			gpuErrorcheck( cudaDeviceSynchronize() );
			gpuErrorcheck( cudaPeekAtLastError() );
		} while (!(*finished));
	} else if (arguments.variant == ASYNC_PUSH_TD) {
		do
		{
			itr++;
			*finished = true;

			sswp::async_push_td<<< num_blocks , num_threads >>>(vGraph.numParts, 
														vGraph.nodePointer,
														vGraph.partNodePointer,
														vGraph.edgeList, 
														dist, 
														finished);

			
			gpuErrorcheck( cudaDeviceSynchronize() );
			gpuErrorcheck( cudaPeekAtLastError() );	
		} while (!(*finished));
	} else if (arguments.variant == SYNC_PUSH_TD) {
		do
		{
			itr++;

			*finished = true;

			sswp::sync_push_td<<< num_blocks , num_threads >>>(vGraph.numParts, 
														vGraph.nodePointer,
														vGraph.partNodePointer,
														vGraph.edgeList, 
														dist, 
														finished,
														(itr % 2 == 1) ? true : false);
			
			gpuErrorcheck( cudaDeviceSynchronize() );
			gpuErrorcheck( cudaPeekAtLastError() );
		} while (!(*finished));
	} else if (arguments.variant == ASYNC_PUSH_DD) {
		do
		{
			itr++;
			*finished = true;

			sswp::async_push_dd<<< num_blocks , num_threads >>>(vGraph.numParts, 
														vGraph.nodePointer,
														vGraph.partNodePointer,
														vGraph.edgeList, 
														dist, 
														finished,
														(itr%2==1) ? label1 : label2,
														(itr%2==1) ? label2 : label1);
			gpuErrorcheck( cudaDeviceSynchronize() );
			gpuErrorcheck( cudaPeekAtLastError() );
		} while (!(*finished));
	}

	if (arguments.energy) nvml.log_point();

	float runtime = timer.Finish();
	float total = totalTimer.Finish();
	cout << "Number of iterations = " << itr << endl;
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

	Result result;
	result.time = total;
	result.energy = nvml.get_energy();

	// Run sequential cpu version and print out useful information
	if (arguments.debug) {
		unsigned int* cpu_dist;
		cpu_dist = new unsigned int[num_nodes];

		for(int i=0; i<num_nodes; i++)
		{
			cpu_dist[i] = 0;
		}
		
		cpu_dist[arguments.sourceNode] = DIST_INFINITY;

		sswp::seq_cpu(vGraph, cpu_dist);

		if (num_nodes < 30) {
			utilities::PrintResults(cpu_dist, num_nodes);
			utilities::PrintResults(dist, num_nodes);
		} else {
			utilities::PrintResults(cpu_dist, 30);
			utilities::PrintResults(dist, 30);
		}

		utilities::CompareArrays(cpu_dist, dist, num_nodes);
	}

	if(arguments.hasOutput)
		utilities::SaveResults(arguments.output, dist, num_nodes);

	gpuErrorcheck(cudaFree(dist));
	gpuErrorcheck(cudaFree(label1));
	gpuErrorcheck(cudaFree(label2));
	gpuErrorcheck(cudaFree(finished));
	gpuErrorcheck(cudaFree(vGraph.nodePointer));
	gpuErrorcheck(cudaFree(vGraph.edgeList));
	gpuErrorcheck(cudaFree(vGraph.partNodePointer));

	return result;
}

Result eegraph_sssp(ArgumentParser &arguments, Graph &graph) {
	// Energy structures initilization
	// Two cpu threads are used to coordinate energy consumption by chanding common flags in nvmlClass
	vector<thread> cpu_threads;
	nvmlClass nvml(arguments.deviceID, arguments.energyFile, arguments.energyStats, to_string(arguments.variant));

	if (arguments.energy) {
		cout << "Starting energy measurements. Timing information will be affected..." << endl;

		cpu_threads.emplace_back(std::thread(&nvmlClass::getStats, &nvml));

  		nvml.log_start();
	}

	VirtualGraph vGraph(graph);
	vGraph.MakeGraph();

	uint num_nodes = graph.num_nodes;
	uint num_edges = graph.num_edges;

	if(arguments.hasDeviceID)
		gpuErrorcheck(cudaSetDevice(arguments.deviceID));

	cudaFree(0);

	unsigned int *dist;
	dist  = new unsigned int[num_nodes];

	bool *label1;
	bool *label2;
	label1 = new bool[num_nodes];
	label2 = new bool[num_nodes];
	
	Timer totalTimer;
	totalTimer.Start();
	if (arguments.energy) nvml.log_point();

	for(int i=0; i<num_nodes; i++)
	{
		dist[i] = DIST_INFINITY;

		if (arguments.variant == ASYNC_PUSH_DD)	label1[i] = true;
		else label1[i]=false;

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
	bool *d_finished;

	gpuErrorcheck(cudaMalloc(&d_nodePointer, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_edgeList, (2*num_edges + num_nodes) * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_dist, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label1, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label2, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_partNodePointer, vGraph.numParts * sizeof(PartPointer)));

	gpuErrorcheck(cudaMemcpy(d_nodePointer, vGraph.nodePointer, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_edgeList, vGraph.edgeList, (2*num_edges + num_nodes) * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_dist, dist, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_label1, label1, num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_label2, label2, num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_partNodePointer, vGraph.partNodePointer, vGraph.numParts * sizeof(PartPointer), cudaMemcpyHostToDevice));

	// Algorithm control variable declarations
	Timer timer;
	int itr = 0;
	int num_threads = 512;
	int num_blocks = vGraph.numParts / num_threads + 1;

	timer.Start();
	if (arguments.energy) nvml.log_point();

	if (arguments.variant == SYNC_PUSH_DD) {
		do
		{
			itr++;
			finished = true;
			gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));
			if(itr % 2 == 1)
			{
				sssp::sync_push_dd<<< num_blocks , num_threads >>>(vGraph.numParts, 
															d_nodePointer,
															d_partNodePointer,
															d_edgeList, 
															d_dist, 
															d_finished,
															d_label1,
															d_label2);
				clearLabel<<< num_blocks , num_threads >>>(d_label1, num_nodes);
			}
			else
			{
				sssp::sync_push_dd<<< num_blocks , num_threads >>>(vGraph.numParts, 
															d_nodePointer, 
															d_partNodePointer,
															d_edgeList, 
															d_dist, 
															d_finished,
															d_label2,
															d_label1);
				clearLabel<<< num_blocks , num_threads >>>(d_label2, num_nodes);
			}

			gpuErrorcheck( cudaDeviceSynchronize() );
			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));

		} while (!(finished));
	} else if (arguments.variant == ASYNC_PUSH_TD) {
		do
		{
			itr++;
			finished = true;
			gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

			sssp::async_push_td<<< num_blocks , num_threads >>>(vGraph.numParts, 
																d_nodePointer,
																d_partNodePointer,
																d_edgeList, 
																d_dist, 
																d_finished);

			gpuErrorcheck( cudaDeviceSynchronize() );
			gpuErrorcheck( cudaPeekAtLastError() );	
			gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));

		} while (!(finished));
	} else if (arguments.variant == SYNC_PUSH_TD) {
		do
		{
			itr++;
			finished = true;
			gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

			sssp::sync_push_td<<< num_blocks , num_threads >>>(vGraph.numParts, 
														d_nodePointer,
														d_partNodePointer,
														d_edgeList, 
														d_dist, 
														d_finished,
														(itr % 2 == 1) ? true : false);
			
			gpuErrorcheck( cudaDeviceSynchronize() );	
			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));

		} while (!(finished));
	} else if (arguments.variant == ASYNC_PUSH_DD) {
		do
		{
			itr++;
			finished = true;
			gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

			sssp::async_push_dd<<< num_blocks , num_threads >>>(vGraph.numParts, 
														d_nodePointer,
														d_partNodePointer,
														d_edgeList, 
														d_dist, 
														d_finished,
														(itr%2==1) ? d_label1 : d_label2,
														(itr%2==1) ? d_label2 : d_label1);

			gpuErrorcheck( cudaDeviceSynchronize() );
			gpuErrorcheck( cudaPeekAtLastError() );
			gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));

		} while (!(finished));
	}

	if (arguments.energy) nvml.log_point();
	gpuErrorcheck(cudaMemcpy(dist, d_dist, num_nodes*sizeof(unsigned int), cudaMemcpyDeviceToHost));
	if (arguments.energy) nvml.log_point();

	float runtime = timer.Finish();
	float total = totalTimer.Finish();
	cout << "Number of iterations = " << itr << endl;
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

	Result result;
	result.time = total;
	result.energy = nvml.get_energy();

	// Run sequential cpu version and print out useful information
	if (arguments.debug) {
		unsigned int* cpu_dist;
		cpu_dist = new unsigned int[num_nodes];

		for(int i=0; i<num_nodes; i++)
		{
			cpu_dist[i] = DIST_INFINITY;
		}
		
		cpu_dist[arguments.sourceNode] = 0;

		//sssp::seq_cpu(graph.edges, graph.weights, num_edges, cpu_dist);
		sssp::seq_cpu(vGraph, cpu_dist);

		utilities::PrintResults(dist, min(30, num_nodes));

		utilities::CompareArrays(cpu_dist, dist, num_nodes);
	}

	if(arguments.hasOutput)
		utilities::SaveResults(arguments.output, dist, num_nodes);

	gpuErrorcheck(cudaFree(d_nodePointer));
	gpuErrorcheck(cudaFree(d_edgeList));
	gpuErrorcheck(cudaFree(d_dist));
	gpuErrorcheck(cudaFree(d_finished));
	gpuErrorcheck(cudaFree(d_label1));
	gpuErrorcheck(cudaFree(d_label2));
	gpuErrorcheck(cudaFree(d_partNodePointer));

	return result;
}

Result eegraph_sssp_um(ArgumentParser &arguments, UMGraph &graph) {	
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

	UMVirtualGraph vGraph(graph);
	vGraph.MakeGraph();

	uint num_nodes = graph.num_nodes;
	uint num_edges = graph.num_edges;

	if(arguments.hasDeviceID)
		gpuErrorcheck(cudaSetDevice(arguments.deviceID));

	cudaFree(0);

	unsigned int *dist;
	bool *label1;
	bool *label2;

	gpuErrorcheck(cudaMallocManaged(&dist, sizeof(unsigned int) * num_nodes));
	gpuErrorcheck(cudaMallocManaged(&label1, sizeof(bool) * num_nodes));
	gpuErrorcheck(cudaMallocManaged(&label2, sizeof(bool) * num_nodes));
	
	Timer totalTimer;
	totalTimer.Start();
	if (arguments.energy) nvml.log_point();

	for(int i=0; i<num_nodes; i++)
	{
		dist[i] = DIST_INFINITY;

		if (arguments.variant == ASYNC_PUSH_DD)	label1[i] = true;
		else label1[i]=false;

		label2[i] = false;
	}
	
	dist[arguments.sourceNode] = 0;
	label1[arguments.sourceNode] = true;

	bool *finished;

	gpuErrorcheck(cudaMallocManaged(&finished, sizeof(bool)));

	// Tell GPU this data is mostly read
	gpuErrorcheck(cudaMemAdvise(vGraph.nodePointer, num_nodes * sizeof(unsigned int), cudaMemAdviseSetReadMostly, arguments.deviceID));
	gpuErrorcheck(cudaMemAdvise(vGraph.edgeList, (2*num_edges + num_nodes) * sizeof(unsigned int), cudaMemAdviseSetReadMostly, arguments.deviceID));
	gpuErrorcheck(cudaMemAdvise(vGraph.partNodePointer, vGraph.numParts * sizeof(PartPointer), cudaMemAdviseSetReadMostly, arguments.deviceID));

	// Algorithm control variable declarations
	Timer timer;
	int itr = 0;
	int num_threads = 512;
	int num_blocks = vGraph.numParts / num_threads + 1;

	timer.Start();
	if (arguments.energy) nvml.log_point();

	if (arguments.variant == SYNC_PUSH_DD) {
		do
		{
			itr++;
			*finished = true;

			if(itr % 2 == 1)
			{
				sssp::sync_push_dd<<< num_blocks , num_threads >>>(vGraph.numParts, 
															vGraph.nodePointer,
															vGraph.partNodePointer,
															vGraph.edgeList, 
															dist, 
															finished,
															label1,
															label2);
				clearLabel<<< num_blocks , num_threads >>>(label1, num_nodes);
			}
			else
			{
				sssp::sync_push_dd<<< num_blocks , num_threads >>>(vGraph.numParts, 
															vGraph.nodePointer, 
															vGraph.partNodePointer,
															vGraph.edgeList, 
															dist, 
															finished,
															label2,
															label1);
				clearLabel<<< num_blocks , num_threads >>>(label2, num_nodes);
			}

			gpuErrorcheck( cudaDeviceSynchronize() );
			gpuErrorcheck( cudaPeekAtLastError() );

		} while (!(*finished));
	} else if (arguments.variant == ASYNC_PUSH_TD) {
		do
		{
			itr++;
			*finished = true;

			sssp::async_push_td<<< num_blocks , num_threads >>>(vGraph.numParts, 
														vGraph.nodePointer,
														vGraph.partNodePointer,
														vGraph.edgeList, 
														dist, 
														finished);

			gpuErrorcheck( cudaDeviceSynchronize() );
			gpuErrorcheck( cudaPeekAtLastError() );

		} while (!(*finished));
	} else if (arguments.variant == SYNC_PUSH_TD) {
		do
		{
			itr++;

			*finished = true;

			sssp::sync_push_td<<< num_blocks , num_threads >>>(vGraph.numParts, 
														vGraph.nodePointer,
														vGraph.partNodePointer,
														vGraph.edgeList, 
														dist, 
														finished,
														(itr % 2 == 1) ? true : false);
			
			gpuErrorcheck( cudaDeviceSynchronize() );
			gpuErrorcheck( cudaPeekAtLastError() );

		} while (!(*finished));
	} else if (arguments.variant == ASYNC_PUSH_DD) {
		do
		{
			itr++;
			*finished = true;

			sssp::async_push_dd<<< num_blocks , num_threads >>>(vGraph.numParts, 
														vGraph.nodePointer,
														vGraph.partNodePointer,
														vGraph.edgeList, 
														dist, 
														finished,
														(itr%2==1) ? label1 : label2,
														(itr%2==1) ? label2 : label1);
			gpuErrorcheck( cudaDeviceSynchronize() );
			gpuErrorcheck( cudaPeekAtLastError() );

		} while (!(*finished));
	}

	if (arguments.energy) nvml.log_point();

	float runtime = timer.Finish();
	float total = totalTimer.Finish();
	cout << "Number of iterations = " << itr << endl;
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

	Result result;
	result.time = total;
	result.energy = nvml.get_energy();

	// Run sequential cpu version and print out useful information
	if (arguments.debug) {
		unsigned int* cpu_dist;
		cpu_dist = new unsigned int[num_nodes];

		for(int i=0; i<num_nodes; i++)
		{
			cpu_dist[i] = DIST_INFINITY;
		}
		
		cpu_dist[arguments.sourceNode] = 0;

		sssp::seq_cpu(vGraph, cpu_dist);

		if (num_nodes < 30) {
			utilities::PrintResults(cpu_dist, num_nodes);
			utilities::PrintResults(dist, num_nodes);
		} else {
			utilities::PrintResults(cpu_dist, 30);
			utilities::PrintResults(dist, 30);
		}

		utilities::CompareArrays(cpu_dist, dist, num_nodes);
	}

	if(arguments.hasOutput)
		utilities::SaveResults(arguments.output, dist, num_nodes);

	gpuErrorcheck(cudaFree(dist));
	gpuErrorcheck(cudaFree(label1));
	gpuErrorcheck(cudaFree(label2));
	gpuErrorcheck(cudaFree(finished));
	gpuErrorcheck(cudaFree(vGraph.nodePointer));
	gpuErrorcheck(cudaFree(vGraph.edgeList));
	gpuErrorcheck(cudaFree(vGraph.partNodePointer));

	return result;
}