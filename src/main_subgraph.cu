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
    Result result;

    ArgumentParser arguments(argc, argv, true, false);

    Graph graph(arguments.input, true);
    UMGraph um_graph(arguments.input, true);

    for (int j = 0; j < num_frameworks; j++) {

        if (j == 0) {
            graph.ReadGraph();
        } else {
            um_graph.ReadGraph();
        }

        VertexSubgraph subgraph(graph, arguments.input, true);
        UMVertexSubgraph um_subgraph(um_graph, arguments.input, true);

        if (j == 0) {

            subgraph.MakeSubgraph(0.05, arguments.sourceNode);
        } else {

            um_subgraph.MakeSubgraph(0.05, arguments.sourceNode);
        }
        cout << endl;

        for (int k = 0; k < num_algorithms; k++) {

            if (j == 0) {
                if (k % num_algorithms == 0) {
                    gpuErrorcheck( cudaDeviceSynchronize() );
                    result = eegraph_bfs(arguments, subgraph);
                    gpuErrorcheck( cudaDeviceSynchronize() );

                    if (arguments.nodeActivity) {
                        cout << "Max active pct:" << utilities::maxActivePct(result.sumLabelsVec, subgraph.subgraph_num_nodes) << endl;
                        cout << "Pct over threshold: " << utilities::pctIterOverThreshold(result.sumLabelsVec, subgraph.subgraph_num_nodes, .50);
                    }
                } else if (k % num_algorithms == 1) {
                    gpuErrorcheck( cudaDeviceSynchronize() );
                    result = eegraph_cc(arguments, subgraph);
                    gpuErrorcheck( cudaDeviceSynchronize() );

                    if (arguments.nodeActivity) {
                        cout << "Max active pct: " << utilities::maxActivePct(result.sumLabelsVec, subgraph.subgraph_num_nodes) << endl;
                        cout << "Pct over threshold: " << utilities::pctIterOverThreshold(result.sumLabelsVec, subgraph.subgraph_num_nodes, .50);
                    }
                } else if (k % num_algorithms == 2) {
                    gpuErrorcheck( cudaDeviceSynchronize() );
                    result = eegraph_pr(arguments, subgraph);
                    gpuErrorcheck( cudaDeviceSynchronize() );

                    if (arguments.nodeActivity) {
                        cout << "Max active pct:" << utilities::maxActivePct(result.sumLabelsVec, subgraph.subgraph_num_nodes) << endl;
                        cout << "Pct over threshold: " << utilities::pctIterOverThreshold(result.sumLabelsVec, subgraph.subgraph_num_nodes, .50);
                    }
                } else if (k % num_algorithms == 3) {
                    gpuErrorcheck( cudaDeviceSynchronize() );
                    result = eegraph_sssp(arguments, subgraph);
                    gpuErrorcheck( cudaDeviceSynchronize() );

                    if (arguments.nodeActivity) {
                        cout << "Max active pct:" << utilities::maxActivePct(result.sumLabelsVec, subgraph.subgraph_num_nodes) << endl;
                        cout << "Pct over threshold: " << utilities::pctIterOverThreshold(result.sumLabelsVec, subgraph.subgraph_num_nodes, .50);
                    }
                } else if (k % num_algorithms == 4) {
                    gpuErrorcheck( cudaDeviceSynchronize() );
                    result = eegraph_sswp(arguments, subgraph);
                    gpuErrorcheck( cudaDeviceSynchronize() );

                    if (arguments.nodeActivity) {
                        cout << "Max active pct:" << utilities::maxActivePct(result.sumLabelsVec, subgraph.subgraph_num_nodes) << endl;
                        cout << "Pct over threshold: " << utilities::pctIterOverThreshold(result.sumLabelsVec, subgraph.subgraph_num_nodes, .50);
                    }
                }
            } else if (j == 1) {
                if (k % num_algorithms == 0) {
                    gpuErrorcheck( cudaDeviceSynchronize() );
                    result = eegraph_bfs_um(arguments, um_subgraph);
                    gpuErrorcheck( cudaDeviceSynchronize() );

                    if (arguments.nodeActivity) {
                        cout << "Max active pct:" << utilities::maxActivePct(result.sumLabelsVec, subgraph.subgraph_num_nodes) << endl;
                        cout << "Pct over threshold: " << utilities::pctIterOverThreshold(result.sumLabelsVec, subgraph.subgraph_num_nodes, .50);
                    }
                } else if (k % num_algorithms == 1) {
                    gpuErrorcheck( cudaDeviceSynchronize() );
                    result = eegraph_cc_um(arguments, um_subgraph);
                    gpuErrorcheck( cudaDeviceSynchronize() );

                    if (arguments.nodeActivity) {
                        cout << "Max active pct:" << utilities::maxActivePct(result.sumLabelsVec, subgraph.subgraph_num_nodes) << endl;
                        cout << "Pct over threshold: " << utilities::pctIterOverThreshold(result.sumLabelsVec, subgraph.subgraph_num_nodes, .50);
                    }
                } else if (k % num_algorithms == 2) {
                    gpuErrorcheck( cudaDeviceSynchronize() );
                    result = eegraph_pr_um(arguments, um_subgraph);
                    gpuErrorcheck( cudaDeviceSynchronize() );

                    if (arguments.nodeActivity) {
                        cout << "Max active pct:" << utilities::maxActivePct(result.sumLabelsVec, subgraph.subgraph_num_nodes) << endl;
                        cout << "Pct over threshold: " << utilities::pctIterOverThreshold(result.sumLabelsVec, subgraph.subgraph_num_nodes, .50);
                    }
                } else if (k % num_algorithms == 3) {
                    gpuErrorcheck( cudaDeviceSynchronize() );
                    result = eegraph_sssp_um(arguments, um_subgraph);
                    gpuErrorcheck( cudaDeviceSynchronize() );

                    if (arguments.nodeActivity) {
                        cout << "Max active pct:" << utilities::maxActivePct(result.sumLabelsVec, subgraph.subgraph_num_nodes) << endl;
                        cout << "Pct over threshold: " << utilities::pctIterOverThreshold(result.sumLabelsVec, subgraph.subgraph_num_nodes, .50);
                    }
                } else if (k % num_algorithms == 4) {
                    gpuErrorcheck( cudaDeviceSynchronize() );
                    result = eegraph_sswp_um(arguments, um_subgraph);
                    gpuErrorcheck( cudaDeviceSynchronize() );

                    if (arguments.nodeActivity) {
                        cout << "Max active pct:" << utilities::maxActivePct(result.sumLabelsVec, subgraph.subgraph_num_nodes) << endl;
                        cout << "Pct over threshold: " << utilities::pctIterOverThreshold(result.sumLabelsVec, subgraph.subgraph_num_nodes, .5000);
                    }
                }
            }
            cout << endl;
        }
    }
}