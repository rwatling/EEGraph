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
#include "../shared/subway_runable.cuh"

#include <iostream>
#include <sys/stat.h>
#include <cstdlib>
#include <unistd.h>

int main (int argc, char** argv) {
    
    const int num_benchmarks = 6;
    //const int num_frameworks = 1;
    const int num_algorithms = 5;
    const int num_trials = 4;

    string benchmarks[num_benchmarks] = {"../datasets/Google/web-Google-trimmed.txt", 
                                        "../datasets/LiveJournal/soc-LiveJournal1-trimmed.txt", 
                                        "../datasets/Orkut/orkut-trimmed.el", 
                                        "../datasets/Pokec/soc-pokec-relationships.txt", 
                                        "../datasets/Road/roadNet-CA-trimmed.txt", 
                                        "../datasets/Skitter/as-skitter-trimmed.txt"}; //Dropped description headers for trimmed files
    string benchnames[num_benchmarks] = {"google", "livejournal", "orkut", "pokec", "road", "skitter"};
    string algorithms[num_algorithms] = {"bfs", "cc", "pr", "sssp", "sswp"};

    string currentBench;
    string currentFramework = "subway";
    string currentAlg;
    string currentVariant;

    SubwayArgumentParser arguments(argc, argv, true, false);

    currentBench = benchnames[arguments.benchmark];
    arguments.input = benchmarks[arguments.benchmark];

    for (int i = 0; i < num_algorithms * 2; i++) {  
        currentAlg = algorithms[i/2];
        if (i % 2 == 1) { arguments.energy = true; }
         
        string trialDir;

        if (!arguments.energy) {
            trialDir = "./" + currentAlg + "/" + currentFramework + "/" + currentBench + "/";
        } else {
            trialDir = "./" + currentAlg + "/" + currentFramework + "-energy/" + currentBench + "/";
        }

        system(("mkdir -p " + trialDir).c_str());

        SubwayGraph<OutEdge> bfs_cc_graph(arguments.input, false);
        GraphPR<OutEdge> pr_graph(arguments.input, true);
        SubwayGraph<OutEdgeWeighted> sssp_sswp_graph(arguments.input, true);

        if (i == 0) {
            bfs_cc_graph.ReadGraph();
        } else if (i == 4) {
            pr_graph.ReadGraph();
        } else if (i == 6) {
            sssp_sswp_graph.ReadGraph();
        }

        //energy

        //number of trials

        //redirect output

        //Free things
        /*if (i == 3) {
            delete bfs_cc_graph.nodePointer;
            cudaFree(bfs_cc_graph.edgeList);
            delete bfs_cc_graph.outDegree;
            delete bfs_cc_graph.label1;
            delete bfs_cc_graph.label2;
            delete bfs_cc_graph.value;
            cudaFree(bfs_cc_graph.d_outDegree);
            cudaFree(bfs_cc_graph.d_value);
            cudaFree(bfs_cc_graph.d_label1);
            cudaFree(bfs_cc_graph.d_label2);
        } else if (i == 5) {
            delete pr_graph.nodePointer;
            cudaFree(pr_graph.edgeList);
            delete pr_graph.outDegree;
            delete pr_graph.value;
            delete pr_graph.delta;
            cudaFree(pr_graph.d_outDegree);
            cudaFree(pr_graph.d_value);
            cudaFree(pr_graph.d_delta);
        } else if (i == 9) {
            delete sssp_sswp_graph.nodePointer;
            cudaFree(sssp_sswp_graph.edgeList);
            delete sssp_sswp_graph.outDegree;
            delete sssp_sswp_graph.label1;
            delete sssp_sswp_graph.label2;
            delete sssp_sswp_graph.value;
            cudaFree(sssp_sswp_graph.d_outDegree);
            cudaFree(sssp_sswp_graph.d_value);
            cudaFree(sssp_sswp_graph.d_label1);
            cudaFree(sssp_sswp_graph.d_label2);
        }*/
    }
}