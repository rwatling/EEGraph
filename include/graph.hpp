#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <iostream>
#include <stdlib.h>
#include <ctime>
#include <fstream>
#include <string>
#include <ctime>
#include <random>
#include <stdio.h>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <math.h>
#include <chrono>

#include "globals.hpp"
#include "timer.hpp"

typedef unsigned int uint;
using namespace std;

class Graph
{
private:

public:
	string graphFilePath;
    string graphFormat;

	bool isWeighted;
	bool hasZeroID;
    bool graphLoaded;

	uint num_nodes;
	uint num_edges;

    vector<Edge> edges;
    vector<uint> weights;

    Graph(string graphFilePath, bool isWeighted);

    void ReadGraph();
    string getFileExtension(string fileName);
};

class VertexSubgraph : public Graph
{
    public:
        Graph* parentGraph;
        float pct;
        bool* selected;

        uint subgraph_num_nodes; // To allow for passing into VirtualGraph

        VertexSubgraph(Graph &graph, string graphFilePath, bool isWeighted);
        void MakeSubgraph(float pct, int sourceNode, time_t seed);
};

#endif	//	GRAPH_HPP
