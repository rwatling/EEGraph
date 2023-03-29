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

int main(int argc, char** argv) {
    const int num_algorithms = 5;
    const int num_frameworks = 2;
	const int pct = 0.05;
    string algorithms[num_algorithms] = {"bfs", "cc", "pr", "sssp", "sswp"};

	string estats = "estats";
	string efile = "efile";
	Result result;
    ArgumentParser arguments(argc, argv, true, false);
    Graph graph(arguments.input, true);
    UMGraph um_graph(arguments.input, true);

	time_t seed = time(NULL);
	for (int j = 0; j < num_frameworks; j++) {
		if (j == 0) {
			graph.ReadGraph();
		} else {
			um_graph.ReadGraph();
		}

		VertexSubgraph subgraph(graph, arguments.input, true);
		UMVertexSubgraph um_subgraph(um_graph, arguments.input, true);

		if (j == 0) {
			subgraph.MakeSubgraph(pct, arguments.sourceNode, seed);
		} else {

			um_subgraph.MakeSubgraph(pct, arguments.sourceNode, seed);
		}
		cout << endl;

		for (int k = 0; k < num_algorithms; k++) {
			cout << "---------------------" << algorithms[k] << "---------------------" << endl;
			if (j == 0) {
				for (int l = 0; l < 4; l++) {
					if(l == 0) {
						cout << "---ASYNC_PUSH_DD---" << endl;
						arguments.variant = ASYNC_PUSH_DD;
					} else if (l == 1) {
						cout << "---ASYNC_PUSH_TD---" << endl;
						arguments.variant = ASYNC_PUSH_TD;
					} else if (l == 2) {
						cout << "---SYNC_PUSH_DD---" << endl;
						arguments.variant = SYNC_PUSH_DD;
					} else if (l == 3) {
						cout << "---SYNC_PUSH_TD---" << endl;
						arguments.variant = SYNC_PUSH_TD;
					}

					if (k % num_algorithms == 0) {
						gpuErrorcheck( cudaDeviceSynchronize() );
						result = eegraph_bfs(arguments, subgraph);
						gpuErrorcheck( cudaDeviceSynchronize() );

						if (arguments.energy) {
							cout << "Energy:" << result.energy << endl;
						}

						if (arguments.nodeActivity) {
							cout << "Max active pct:" << utilities::maxActivePct(result.sumLabelsVec, subgraph.subgraph_num_nodes) << endl;
							cout << "Pct over threshold: " << utilities::pctIterOverThreshold(result.sumLabelsVec, subgraph.subgraph_num_nodes, .50);
						}
					} else if (k % num_algorithms == 1) {
						gpuErrorcheck( cudaDeviceSynchronize() );
						result = eegraph_cc(arguments, subgraph);
						gpuErrorcheck( cudaDeviceSynchronize() );

						if (arguments.energy) {
							cout << "Energy:" << result.energy << endl;
						}

						if (arguments.nodeActivity) {
							cout << "Max active pct: " << utilities::maxActivePct(result.sumLabelsVec, subgraph.subgraph_num_nodes) << endl;
							cout << "Pct over threshold: " << utilities::pctIterOverThreshold(result.sumLabelsVec, subgraph.subgraph_num_nodes, .50);
						}
					} else if (k % num_algorithms == 2) {
						gpuErrorcheck( cudaDeviceSynchronize() );
						result = eegraph_pr(arguments, subgraph);
						gpuErrorcheck( cudaDeviceSynchronize() );

						if (arguments.energy) {
							cout << "Energy:" << result.energy << endl;
						}

						if (arguments.nodeActivity) {
							cout << "Max active pct:" << utilities::maxActivePct(result.sumLabelsVec, subgraph.subgraph_num_nodes) << endl;
							cout << "Pct over threshold: " << utilities::pctIterOverThreshold(result.sumLabelsVec, subgraph.subgraph_num_nodes, .50);
						}
					} else if (k % num_algorithms == 3) {
						gpuErrorcheck( cudaDeviceSynchronize() );
						result = eegraph_sssp(arguments, subgraph);
						gpuErrorcheck( cudaDeviceSynchronize() );

						if (arguments.energy) {
							cout << "Energy:" << result.energy << endl;
						}

						if (arguments.nodeActivity) {
							cout << "Max active pct:" << utilities::maxActivePct(result.sumLabelsVec, subgraph.subgraph_num_nodes) << endl;
							cout << "Pct over threshold: " << utilities::pctIterOverThreshold(result.sumLabelsVec, subgraph.subgraph_num_nodes, .50);
						}
					} else if (k % num_algorithms == 4) {
						gpuErrorcheck( cudaDeviceSynchronize() );
						result = eegraph_sswp(arguments, subgraph);
						gpuErrorcheck( cudaDeviceSynchronize() );

						if (arguments.energy) {
							cout << "Energy:" << result.energy << endl;
						}

						if (arguments.nodeActivity) {
							cout << "Max active pct:" << utilities::maxActivePct(result.sumLabelsVec, subgraph.subgraph_num_nodes) << endl;
							cout << "Pct over threshold: " << utilities::pctIterOverThreshold(result.sumLabelsVec, subgraph.subgraph_num_nodes, .50);
						}
					}
				}
			} else if (j == 1) {
				for (int l = 0; l < 4; l++) {
					if(l == 0) {
						cout << "---UM-ASYNC_PUSH_DD---" << endl;
						arguments.variant = ASYNC_PUSH_DD;
					} else if (l == 1) {
						cout << "---UM-ASYNC_PUSH_TD---" << endl;
						arguments.variant = ASYNC_PUSH_TD;
					} else if (l == 2) {
						cout << "---UM-SYNC_PUSH_DD---" << endl;
						arguments.variant = SYNC_PUSH_DD;
					} else if (l == 3) {
						cout << "---UM-SYNC_PUSH_TD---" << endl;
						arguments.variant = SYNC_PUSH_TD;
					}
					if (k % num_algorithms == 0) {
						gpuErrorcheck( cudaDeviceSynchronize() );
						result = eegraph_bfs_um(arguments, um_subgraph);
						gpuErrorcheck( cudaDeviceSynchronize() );

						if (arguments.energy) {
							cout << "Energy:" << result.energy << endl;
						}

						if (arguments.nodeActivity) {
							cout << "Max active pct:" << utilities::maxActivePct(result.sumLabelsVec, subgraph.subgraph_num_nodes) << endl;
							cout << "Pct over threshold: " << utilities::pctIterOverThreshold(result.sumLabelsVec, subgraph.subgraph_num_nodes, .50);
						}
					} else if (k % num_algorithms == 1) {
						gpuErrorcheck( cudaDeviceSynchronize() );
						result = eegraph_cc_um(arguments, um_subgraph);
						gpuErrorcheck( cudaDeviceSynchronize() );

						if (arguments.energy) {
							cout << "Energy:" << result.energy << endl;
						}

						if (arguments.nodeActivity) {
							cout << "Max active pct:" << utilities::maxActivePct(result.sumLabelsVec, subgraph.subgraph_num_nodes) << endl;
							cout << "Pct over threshold: " << utilities::pctIterOverThreshold(result.sumLabelsVec, subgraph.subgraph_num_nodes, .50);
						}
					} else if (k % num_algorithms == 2) {
						gpuErrorcheck( cudaDeviceSynchronize() );
						result = eegraph_pr_um(arguments, um_subgraph);
						gpuErrorcheck( cudaDeviceSynchronize() );

						if (arguments.energy) {
							cout << "Energy:" << result.energy << endl;
						}

						if (arguments.nodeActivity) {
							cout << "Max active pct:" << utilities::maxActivePct(result.sumLabelsVec, subgraph.subgraph_num_nodes) << endl;
							cout << "Pct over threshold: " << utilities::pctIterOverThreshold(result.sumLabelsVec, subgraph.subgraph_num_nodes, .50);
						}
					} else if (k % num_algorithms == 3) {
						gpuErrorcheck( cudaDeviceSynchronize() );
						result = eegraph_sssp_um(arguments, um_subgraph);
						gpuErrorcheck( cudaDeviceSynchronize() );

						if (arguments.energy) {
							cout << "Energy:" << result.energy << endl;
						}

						if (arguments.nodeActivity) {
							cout << "Max active pct:" << utilities::maxActivePct(result.sumLabelsVec, subgraph.subgraph_num_nodes) << endl;
							cout << "Pct over threshold: " << utilities::pctIterOverThreshold(result.sumLabelsVec, subgraph.subgraph_num_nodes, .50);
						}
					} else if (k % num_algorithms == 4) {
						gpuErrorcheck( cudaDeviceSynchronize() );
						result = eegraph_sswp_um(arguments, um_subgraph);
						gpuErrorcheck( cudaDeviceSynchronize() );

						if (arguments.energy) {
							cout << "Energy:" << result.energy << endl;
						}

						if (arguments.nodeActivity) {
							cout << "Max active pct:" << utilities::maxActivePct(result.sumLabelsVec, subgraph.subgraph_num_nodes) << endl;
							cout << "Pct over threshold: " << utilities::pctIterOverThreshold(result.sumLabelsVec, subgraph.subgraph_num_nodes, .5000);
						}
					}
				}
			}
			cout << endl;
		}
	}
}