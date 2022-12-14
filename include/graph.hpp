#ifndef GRAPH_HPP
#define GRAPH_HPP

#define RAND_SEED 123456
#define RAND_RANGE 256

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

#endif	//	GRAPH_HPP



