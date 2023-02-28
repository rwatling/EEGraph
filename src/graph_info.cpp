#include "../include/graph.hpp"
#include "../include/argument_parsing.hpp"

int main(int argc, char** argv) {
    const int num_benchmarks = 5;
    string benchmarks[num_benchmarks] = {"../datasets/Google/web-Google-trimmed.txt", 
                                        "../datasets/LiveJournal/soc-LiveJournal1-trimmed.txt",
                                        "../datasets/Road/roadNet-CA-trimmed.txt", 
                                        "../datasets/Skitter/as-skitter-trimmed.txt",
										"../datasets/Wiki/wiki-Talk-trimmed.txt"};

    ArgumentParser arguments(argc, argv, true, false);
    unsigned int* degree;
    unsigned int max;

    for (int i = 0; i < num_benchmarks; i++) {
        arguments.input = benchmarks[i];
	    Graph graph(arguments.input, true);
        graph.ReadGraph();

        degree = new unsigned int[graph.num_nodes];
        max = 0;

        for (unsigned int j = 0; j < graph.num_edges; j++) {
            degree[graph.edges[j].source] = 0;
        }

        for (unsigned int j = 0; j < graph.num_edges; j++) {
            degree[graph.edges[j].source]++;
            unsigned int temp = degree[graph.edges[j].source];
            if (temp > max) {
                max = temp;
            }
        }

        cout << "Max degree: " << max << endl;

        unsigned int cut = .20 * graph.num_nodes;
        unsigned int sum = 0;

        //cout << "Cut: " << cut << endl;

        for (unsigned int j = 0; j < cut; j++) {
            sum += degree[j];
        }

        cout << "Avg Degree First 20 percent of Vertices: " << (float) sum / (float) cut << endl;
        sum = 0;

        for (unsigned int j = cut; j < graph.num_nodes; j++) {
            sum += degree[j];
        }

        cout << "Avg Degree Remaining 80 percent of Vertices: " << (float) sum / (float) (graph.num_nodes - cut) << endl;
    }

    return 0;
}