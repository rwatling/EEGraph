#include "../include/graph.hpp"
#include "../include/argument_parsing.hpp"
#include "../include/gpu_error_check.cuh"
#include "../include/cuda_includes.cuh"
#include "../include/nvmlClass.cuh"
#include "../include/gpu_utils.cuh"
#include "../include/um_virtual_graph.cuh"
#include "../include/um_graph.cuh"

#include "../Subway/shared/subway_runable.cuh"
#include "../Subway/shared/subway_graph.cuh"
#include "../Subway/shared/subway_argument_parsing.cuh"
#include "../Subway/shared/subway_globals.hpp"
#include "../Subway/shared/subgraph.cuh"
#include "../Subway/shared/partitioner.cuh"
#include "../Subway/shared/gpu_kernels.cuh"

#include <iostream>
#include <sys/stat.h>

int main (int argc, char** argv) {
    
    const int num_benchmarks = 6;
    const int num_frameworks = 3;
    const int num_algorithms = 5;
    const int num_trials = 3;
    const int num_variants = 4;

    string benchmarks[num_benchmarks] = {"../datasets/Google/web-Google.txt", 
                                        "../datasets/LiveJournal/soc-LiveJournal1.txt", 
                                        "../datasets/Orkut/orkut.el", 
                                        "../datasets/Pokec/soc-Pokec-relationships.txt", 
                                        "../datasets/Road/roadNet-CA.txt", 
                                        "../datasets/Skitter/as-skitter.txt"};
    string benchnames[num_benchmarks] = {"google", "livejournal", "orkut", "pokec", "road", "skitter"};
    string frameworks[num_frameworks] = {"classic", "um", "subway"};
    string algorithms[num_algorithms] = {"bfs", "cc", "pr", "sssp", "sswp"};

    string currentBench;
    string currentFramework;
    string currentAlg;
    string currentVariant;

    ArgumentParser arguments(argc, argv, true, false);
    SubwayArgumentParser subway_args(argc, argv, true, false);

    // Initialize graphs
	/*Graph graph(arguments.input, true);
	graph.ReadGraph();

    graph.~Graph();

    SubwayGraph<OutEdgeWeighted> sssp_sswp_graph(arguments.input, true);
    sssp_sswp_graph.ReadGraph();

    GraphPR<OutEdge> pr_graph(arguments.input, true);
	pr_graph.ReadGraph();

    SubwayGraph<OutEdge> bfs_cc_graph(arguments.input, false);
	bfs_cc_graph.ReadGraph();*/

    for (int i = 0; i < num_benchmarks; i++) {
        currentBench = benchnames[i];

        for (int j = 0; j < num_frameworks; j++) {
            currentFramework = frameworks[j];

            for (int k = 0; k < num_algorithms; k++) {
                currentAlg = algorithms[k];

                string trialDir = "./" + currentAlg + "/" + currentFramework + "/" + currentBench + "/";
                cout << trialDir << endl;

                //make directory

                for (int l = 0; l < num_trials; l++) {

                    //filename = trialnumber-variant

                    //redirect cout

                    //if arguments.energy

                    if (j == 0 || j == 1) {
                        for (int m = 0; m < num_variants; m++) {
                            if (m == 0) {
                                arguments.variant = ASYNC_PUSH_TD;
                                currentVariant = "async-push-td";
                            } else if (m == 1) {
                                arguments.variant = ASYNC_PUSH_DD;
                                currentVariant = "async-push-dd";
                            } else if (m == 2) {
                                arguments.variant = SYNC_PUSH_TD;
                                currentVariant = "sync-push-td";
                            } else if (m == 3) {
                                arguments.variant = SYNC_PUSH_DD;
                                currentVariant = "sync-push-dd";
                            }
                        }
                    } else if (j == 2) {
                        for (int m = 0; m < 2; m++) {
                            if (m == 0) {
                                currentVariant = "async";
                            } else if (m == 1) {
                                currentVariant = "sync";
                            }
                        }
                    }
                }
            }
        }
    }
}