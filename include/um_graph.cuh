#ifndef UM_GRAPH_CUH
#define UM_GRAPH_CUH

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

class UMGraph
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

    Edge* edges;
    uint* weights;

    UMGraph(string graphFilePath, bool isWeighted);

    void ReadGraph();
    string getFileExtension(string fileName);
};

#endif	//	UM_GRAPH_CUH



