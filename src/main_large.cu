#include "../include/graph.hpp"
#include "../include/argument_parsing.hpp"
#include "../include/gpu_error_check.cuh"
#include "../include/cuda_includes.cuh"
#include "../include/nvmlClass.cuh"
#include "../include/gpu_utils.cuh"
#include "../include/um_virtual_graph.cuh"
#include "../include/um_graph.cuh"
#include "../include/eegraph.cuh"

#include <iostream>
#include <sys/stat.h>
#include <cstdlib>
#include <unistd.h>

int main (int argc, char** argv) {
    const int num_benchmarks = 3;
    const int num_frameworks = 2;
    const int num_algorithms = 5;
    const int num_trials = 3;

    string benchmarks[num_benchmarks] = {"/home/share/graph_data/raw/twitter_mpi/twitter.el",
        								"/home/share/graph_data/raw/friendster_snap/fs.el",
										"/home/share/graph_data/raw/twitter_www/twitter.www.el"};
    string benchnames[num_benchmarks] = {"tw", "fs", "tw2"};
    string frameworks[num_frameworks] = {"classic", "um"};
    string algorithms[num_algorithms] = {"bfs", "cc", "pr", "sssp", "sswp"};

    string currentBench;
    string currentFramework;
    string currentAlg;
    string currentVariant;

    ArgumentParser arguments(argc, argv, true, false);

	for (int i = 2; i < num_benchmarks; i++) {
		currentBench = benchnames[i];

		//Read in graphs
		arguments.input = benchmarks[i];
		Graph graph(arguments.input, true);
		UMGraph um_graph(arguments.input,true);
		gpuErrorcheck( cudaDeviceSynchronize() );

		for (int j = 1; j < num_frameworks; j++) {
			if (j == 0) {
				cout << "---graph---" << endl;
				graph.ReadGraph();
			} else if ( j == 1) {
				cout << "---um graph---" << endl;
				um_graph.ReadGraph();
				gpuErrorcheck( cudaDeviceSynchronize() );
			}

			for (int k = 0; k < num_algorithms * 2; k++) {
				currentAlg = algorithms[k % num_algorithms];
				
				if (k >= num_algorithms) { arguments.energy = true; } 
				else { arguments.energy = false; }

				string trialDir;

				if (!arguments.energy) {
					trialDir = "./" + currentAlg + "-large/" + currentBench + "/";
				} else {
					trialDir = "./" + currentAlg + "-large/" + currentBench + "-energy/";
				}

				system(("mkdir -p " + trialDir).c_str());

				for (int l = 0; l < num_trials; l++) {

					if (j == 0) {   //Classic
						for (int m = 0; m < 4; m++) {
							if (m == 0) {
								arguments.variant = ASYNC_PUSH_TD;
								currentVariant = "async-push-td";
								string filename = trialDir + currentVariant + to_string(l);
								
								if (arguments.energy) {
									arguments.energyFile = trialDir + currentVariant + "-readings" + to_string(l);
									arguments.energyStats = trialDir  + currentVariant + "-stats" + to_string(l);
								}

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
								} else if (k % num_algorithms == 1) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_cc(arguments, graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								} else if (k % num_algorithms == 2) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_pr(arguments, graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								} else if (k % num_algorithms == 3) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_sssp(arguments, graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								} else if (k % num_algorithms == 4) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_sswp(arguments, graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								}
							
								// Redirect cout back to screen
								cout.rdbuf(stream_buffer_cout);                                
								file.close();
							} else if (m == 1) {
								arguments.variant = ASYNC_PUSH_DD;
								currentVariant = "async-push-dd";
								string filename = trialDir + currentVariant + to_string(l);

								if (arguments.energy) {
									arguments.energyFile = trialDir + currentVariant + "-readings" + to_string(l);
									arguments.energyStats = trialDir  + currentVariant + "-stats" + to_string(l);
								}

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
								} else if (k % num_algorithms == 1) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_cc(arguments, graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								} else if (k % num_algorithms == 2) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_pr(arguments, graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								} else if (k % num_algorithms == 3) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_sssp(arguments, graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								} else if (k % num_algorithms == 4) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_sswp(arguments, graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								}
							
								// Redirect cout back to screen
								cout.rdbuf(stream_buffer_cout);                                
								file.close();
							} else if (m == 2) {
								arguments.variant = SYNC_PUSH_TD;
								currentVariant = "sync-push-td";
								string filename = trialDir + currentVariant + to_string(l);
								
								if (arguments.energy) {
									arguments.energyFile = trialDir + currentVariant + "-readings" + to_string(l);
									arguments.energyStats = trialDir  + currentVariant + "-stats" + to_string(l);
								}

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
								} else if (k % num_algorithms == 1) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_cc(arguments, graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								} else if (k % num_algorithms == 2) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_pr(arguments, graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								} else if (k % num_algorithms == 3) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_sssp(arguments, graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								} else if (k % num_algorithms == 4) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_sswp(arguments, graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								}
							
								// Redirect cout back to screen
								cout.rdbuf(stream_buffer_cout);                                
								file.close();
							} else if (m == 3) {
								arguments.variant = SYNC_PUSH_DD;
								currentVariant = "sync-push-dd";
								string filename = trialDir + currentVariant + to_string(l);
								
								if (arguments.energy) {
									arguments.energyFile = trialDir + currentVariant + "-readings" + to_string(l);
									arguments.energyStats = trialDir  + currentVariant + "-stats" + to_string(l);
								}

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
								} else if (k % num_algorithms == 1) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_cc(arguments, graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								} else if (k % num_algorithms == 2) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_pr(arguments, graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								} else if (k % num_algorithms == 3) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_sssp(arguments, graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								} else if (k % num_algorithms == 4) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_sswp(arguments, graph);
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
								
								if (arguments.energy) {
									arguments.energyFile = trialDir + currentVariant + "-readings" + to_string(l);
									arguments.energyStats = trialDir  + currentVariant + "-stats" + to_string(l);
								}

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
								} else if (k % num_algorithms == 1) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_cc_um(arguments, um_graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								} else if (k % num_algorithms == 2) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_pr_um(arguments, um_graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								} else if (k % num_algorithms == 3) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_sssp_um(arguments, um_graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								} else if (k % num_algorithms == 4) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_sswp_um(arguments, um_graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								}
							
								// Redirect cout back to screen
								cout.rdbuf(stream_buffer_cout);                                
								file.close();
							} else if (m == 1) {
								arguments.variant = ASYNC_PUSH_DD;
								currentVariant = "um-async-push-dd";
								string filename = trialDir + currentVariant + to_string(l);

								if (arguments.energy) {
									arguments.energyFile = trialDir + currentVariant + "-readings" + to_string(l);
									arguments.energyStats = trialDir  + currentVariant + "-stats" + to_string(l);
								}

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
								} else if (k % num_algorithms == 1) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_cc_um(arguments, um_graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								} else if (k % num_algorithms == 2) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_pr_um(arguments, um_graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								} else if (k % num_algorithms == 3) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_sssp_um(arguments, um_graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								} else if (k % num_algorithms == 4) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_sswp_um(arguments, um_graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								}
							
								// Redirect cout back to screen
								cout.rdbuf(stream_buffer_cout);                                
								file.close();
							} else if (m == 2) {
								arguments.variant = SYNC_PUSH_TD;
								currentVariant = "um-sync-push-td";
								string filename = trialDir + currentVariant + to_string(l);

								if (arguments.energy) {
									arguments.energyFile = trialDir + currentVariant + "-readings" + to_string(l);
									arguments.energyStats = trialDir  + currentVariant + "-stats" + to_string(l);
								}

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
								} else if (k % num_algorithms == 1) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_cc_um(arguments, um_graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								} else if (k % num_algorithms == 2) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_pr_um(arguments, um_graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								} else if (k % num_algorithms == 3) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_sssp_um(arguments, um_graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								} else if (k % num_algorithms == 4) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_sswp_um(arguments, um_graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								}
							
								// Redirect cout back to screen
								cout.rdbuf(stream_buffer_cout);                                
								file.close();
							} else if (m == 3) {
								arguments.variant = SYNC_PUSH_DD;
								currentVariant = "um-sync-push-dd";
								string filename = trialDir + currentVariant + to_string(l);

								if (arguments.energy) {
									arguments.energyFile = trialDir + currentVariant + "-readings" + to_string(l);
									arguments.energyStats = trialDir  + currentVariant + "-stats" + to_string(l);
								}

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
								} else if (k % num_algorithms == 1) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_cc_um(arguments, um_graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								} else if (k % num_algorithms == 2) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_pr_um(arguments, um_graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								} else if (k % num_algorithms == 3) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_sssp_um(arguments, um_graph);
									gpuErrorcheck( cudaDeviceSynchronize() );
								} else if (k % num_algorithms == 4) {
									gpuErrorcheck( cudaDeviceSynchronize() );
									eegraph_sswp_um(arguments, um_graph);
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