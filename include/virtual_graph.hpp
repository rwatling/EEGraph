#ifndef VIRTUAL_GRAPH_HPP
#define VIRTUAL_GRAPH_HPP

#include "graph.hpp"
#include "globals.hpp"


class VirtualGraph
{
private:

public:
	Graph *graph;
	uint *edgeList;
    uint *nodePointer;
    uint *inDegree;
    uint *outDegree;
    unsigned int numParts;
    PartPointer *partNodePointer;

    vector<uint> weights;
    
	VirtualGraph(Graph &graph);
	void MakeGraph();
	void MakeUGraph();
};


#endif	//	VIRTUAL_GRAPH_HPP
