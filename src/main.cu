#include "../include/graph.hpp"
#include "../include/argument_parsing.hpp"
#include "../include/gpu_error_check.cuh"
#include "../include/cuda_includes.cuh"
#include "../include/nvmlClass.cuh"
#include "../include/gpu_utils.cuh"
#include "../include/um_virtual_graph.cuh"
#include "../include/um_graph.cuh"
#include "../include/eegraph_runable.cuh"

#include <iostream>
#include <sys/stat.h>
#include <cstdlib>
#include <unistd.h>

int main (int argc, char** argv) {
    
    const int num_benchmarks = 5;
    const int num_frameworks = 2;
    const int num_algorithms = 5;
    const int num_trials = 1;

    string benchmarks[num_benchmarks] = {"../datasets/Google/web-Google-trimmed.txt", 
                                        "../datasets/LiveJournal/soc-LiveJournal1-trimmed.txt",
                                        "../datasets/Road/roadNet-CA-trimmed.txt", 
                                        "../datasets/Skitter/as-skitter-trimmed.txt",
										"../datasets/Wiki/wiki-Talk-trimmed.txt"}; //Dropped description headers for trimmed files
    string benchnames[num_benchmarks] = {"google", "LiveJournal", "road", "skitter", "wiki"};
    string frameworks[num_frameworks] = {"classic", "um"};
    string algorithms[num_algorithms] = {"bfs", "cc", "pr", "sssp", "sswp"};

    string currentBench;
    string currentFramework;
    string currentAlg;
    string currentVariant;

    ArgumentParser arguments(argc, argv, true, false);

	for (int i = 0; i < num_benchmarks; i++) {
		currentBench = benchnames[i];

		//Read in graphs
		arguments.input = benchmarks[i];
		Graph graph(arguments.input, true);
		UMGraph um_graph(arguments.input,true);
		gpuErrorcheck( cudaDeviceSynchronize() );

		for (int j = 1; j < num_frameworks; j++) {
			if (j == 0) {
				graph.ReadGraph();
			} else if ( j == 1) {
				um_graph.ReadGraph();
				gpuErrorcheck( cudaDeviceSynchronize() );
			}

			for (int k = 0; k < num_algorithms * 2; k++) {
				currentAlg = algorithms[k % num_algorithms];
				
				if (k >= num_algorithms) { arguments.energy = true; } 
				else { arguments.energy = false; }

				string trialDir;

				if (!arguments.energy) {
					trialDir = "./" + currentAlg + "/" + currentBench + "/";
				} else {
					trialDir = "./" + currentAlg + "/" + currentBench + "-energy/";
				}

				system(("mkdir -p " + trialDir).c_str());

				for (int l = 0; l < num_trials; l++) {

					if (j == 0) {   //Classic
						for (int m = 0; m < 4; m++) {
							if (m == 0) {
								arguments.variant = ASYNC_PUSH_TD;
								currentVariant = "async-push-td";
								string filename = trialDir + currentVariant + to_string(l);
								arguments.energyFile = trialDir + currentVariant + "-readings" + to_string(l);
								arguments.energyStats = trialDir  + currentVariant + "-readings" + to_string(l);

								system(("touch " + filename).c_str());
								fstream file;
								file.open(filename);

								// Backup streambuffers of  cout
								streambuf* stream_buffer_cout = cout.rdbuf();
							
								// Get the streambuffer of the file
								streambuf* stream_buffer_file = file.rdbuf();
							
								// Redirect cout to file
								cout.rdbuf(stream_buffer_file);
							
								if (k % num_algorithms == 0) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_bfs(arguments, graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								}
							
								// Redirect cout back to screen
								cout.rdbuf(stream_buffer_cout);                                
								file.close();
							} else if (m == 1) {
								arguments.variant = ASYNC_PUSH_DD;
								currentVariant = "async-push-dd";
								string filename = trialDir + currentVariant + to_string(l);
								arguments.energyFile = trialDir + currentVariant + "-readings" + to_string(l);
								arguments.energyStats = trialDir  + currentVariant + "-readings" + to_string(l);


								system(("touch " + filename).c_str());
								fstream file;
								file.open(filename);

								// Backup streambuffers of  cout
								streambuf* stream_buffer_cout = cout.rdbuf();
							
								// Get the streambuffer of the file
								streambuf* stream_buffer_file = file.rdbuf();
							
								// Redirect cout to file
								cout.rdbuf(stream_buffer_file);
							
								if (k % num_algorithms == 0) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_bfs(arguments, graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								}
							
								// Redirect cout back to screen
								cout.rdbuf(stream_buffer_cout);                                
								file.close();
							} else if (m == 2) {
								arguments.variant = SYNC_PUSH_TD;
								currentVariant = "sync-push-td";
								string filename = trialDir + currentVariant + to_string(l);
								arguments.energyFile = trialDir + currentVariant + "-readings" + to_string(l);
								arguments.energyStats = trialDir  + currentVariant + "-readings" + to_string(l);

								system(("touch " + filename).c_str());
								fstream file;
								file.open(filename);

								// Backup streambuffers of  cout
								streambuf* stream_buffer_cout = cout.rdbuf();
							
								// Get the streambuffer of the file
								streambuf* stream_buffer_file = file.rdbuf();
							
								// Redirect cout to file
								cout.rdbuf(stream_buffer_file);
							
								if (k % num_algorithms == 0) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_bfs(arguments, graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								}
							
								// Redirect cout back to screen
								cout.rdbuf(stream_buffer_cout);                                
								file.close();
							} else if (m == 3) {
								arguments.variant = SYNC_PUSH_DD;
								currentVariant = "sync-push-dd";
								string filename = trialDir + currentVariant + to_string(l);
								arguments.energyFile = trialDir + currentVariant + "-readings" + to_string(l);
								arguments.energyStats = trialDir  + currentVariant + "-readings" + to_string(l);

								system(("touch " + filename).c_str());
								fstream file;
								file.open(filename);

								// Backup streambuffers of  cout
								streambuf* stream_buffer_cout = cout.rdbuf();
							
								// Get the streambuffer of the file
								streambuf* stream_buffer_file = file.rdbuf();
							
								// Redirect cout to file
								cout.rdbuf(stream_buffer_file);
							
								if (k % num_algorithms == 0) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_bfs(arguments, graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								}
							
								// Redirect cout back to screen
								cout.rdbuf(stream_buffer_cout);                                
								file.close();
							}
						}
					} else if (j == 1) {    //UM
						for (int m = 0; m < 4; m++) {
							if (m == 0) {
								arguments.variant = ASYNC_PUSH_TD;
								currentVariant = "um-async-push-td";
								string filename = trialDir + currentVariant + to_string(l);
								arguments.energyFile = trialDir + currentVariant + "-readings" + to_string(l);
								arguments.energyStats = trialDir  + currentVariant + "-readings" + to_string(l);

								system(("touch " + filename).c_str());
								fstream file;
								file.open(filename);

								// Backup streambuffers of  cout
								streambuf* stream_buffer_cout = cout.rdbuf();
							
								// Get the streambuffer of the file
								streambuf* stream_buffer_file = file.rdbuf();
							
								// Redirect cout to file
								cout.rdbuf(stream_buffer_file);
							
								if (k % num_algorithms == 0) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_bfs_um(arguments, um_graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								}
							
								// Redirect cout back to screen
								cout.rdbuf(stream_buffer_cout);                                
								file.close();
							} else if (m == 1) {
								arguments.variant = ASYNC_PUSH_DD;
								currentVariant = "um-async-push-dd";
								string filename = trialDir + currentVariant + to_string(l);
								arguments.energyFile = trialDir + currentVariant + "-readings" + to_string(l);
								arguments.energyStats = trialDir  + currentVariant + "-readings" + to_string(l);

								system(("touch " + filename).c_str());
								fstream file;
								file.open(filename);

								// Backup streambuffers of  cout
								streambuf* stream_buffer_cout = cout.rdbuf();
							
								// Get the streambuffer of the file
								streambuf* stream_buffer_file = file.rdbuf();
							
								// Redirect cout to file
								cout.rdbuf(stream_buffer_file);
							
								if (k % num_algorithms == 0) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_bfs_um(arguments, um_graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								}
							
								// Redirect cout back to screen
								cout.rdbuf(stream_buffer_cout);                                
								file.close();
							} else if (m == 2) {
								arguments.variant = SYNC_PUSH_TD;
								currentVariant = "um-sync-push-td";
								string filename = trialDir + currentVariant + to_string(l);
								arguments.energyFile = trialDir + currentVariant + "-readings" + to_string(l);
								arguments.energyStats = trialDir  + currentVariant + "-readings" + to_string(l);

								system(("touch " + filename).c_str());
								fstream file;
								file.open(filename);

								// Backup streambuffers of  cout
								streambuf* stream_buffer_cout = cout.rdbuf();
							
								// Get the streambuffer of the file
								streambuf* stream_buffer_file = file.rdbuf();
							
								// Redirect cout to file
								cout.rdbuf(stream_buffer_file);
							
								if (k % num_algorithms == 0) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_bfs_um(arguments, um_graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								}
							
								// Redirect cout back to screen
								cout.rdbuf(stream_buffer_cout);                                
								file.close();
							} else if (m == 3) {
								arguments.variant = SYNC_PUSH_DD;
								currentVariant = "um-sync-push-dd";
								string filename = trialDir + currentVariant + to_string(l);
								arguments.energyFile = trialDir + currentVariant + "-readings" + to_string(l);
								arguments.energyStats = trialDir  + currentVariant + "-readings" + to_string(l);

								system(("touch " + filename).c_str());
								fstream file;
								file.open(filename);

								// Backup streambuffers of  cout
								streambuf* stream_buffer_cout = cout.rdbuf();
							
								// Get the streambuffer of the file
								streambuf* stream_buffer_file = file.rdbuf();
							
								// Redirect cout to file
								cout.rdbuf(stream_buffer_file);
							
								if (k % num_algorithms == 0) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_bfs_um(arguments, um_graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								}
							
								// Redirect cout back to screen
								cout.rdbuf(stream_buffer_cout);                                
								file.close();
							}
						}
					}
				}
			}

			gpuErrorcheck( cudaDeviceSynchronize() );
			if (j == 1) {
				gpuErrorcheck(cudaFree(um_graph.edges));
				gpuErrorcheck(cudaFree(um_graph.weights));
			}
			gpuErrorcheck( cudaDeviceReset() );
			gpuErrorcheck( cudaDeviceSynchronize() );
		}
	}

	return 0;
}

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

	Timer totalTimer;
	totalTimer.Start();
	if (arguments.energy) nvml.log_point();

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

	Timer totalTimer;
	totalTimer.Start();
	if (arguments.energy) nvml.log_point();

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