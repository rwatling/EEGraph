#ifndef SUBWAY_GRAPH_CUH
#define SUBWAY_GRAPH_CUH


#include "subway_globals.hpp"

template <class E>
class SubwayGraph
{
private:

public:
	string graphFilePath;
	bool isWeighted;
	bool isLarge;
	uint num_nodes;
	uint num_edges;
	uint *nodePointer;
	E *edgeList;
	uint *outDegree;
	bool *label1;
	bool *label2;
	uint *value;
	uint *d_outDegree;
	uint *d_value;
	bool *d_label1;
	bool *d_label2;
	string graphFormat;
    SubwayGraph(string graphFilePath, bool isWeighted);
    string GetFileExtension(string fileName);
    void AssignW8(uint w8, uint index);
    void ReadGraph();
};

template <class E>
class GraphPR
{
private:

public:
	string graphFilePath;
	bool isWeighted;
	bool isLarge;
	uint num_nodes;
	uint num_edges;
	uint *nodePointer;
	E *edgeList;
	uint *outDegree;
	float *value;
	float *delta;
	uint *d_outDegree;
	float *d_value;
	float *d_delta;
	string graphFormat;
    GraphPR(string graphFilePath, bool isWeighted);
    string GetFileExtension(string fileName);
    void AssignW8(uint w8, uint index);
    void ReadGraph();
};

#endif	//	GRAPH_CUH



