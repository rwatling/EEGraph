#ifndef UM_VIRTUAL_GRAPH_CUH
#define UM_VIRTUAL_GRAPH_CUH

#include "um_graph.cuh"
#include "globals.hpp"

class UMVirtualGraph
{
private:

public:
	UMGraph *graph;
	uint *edgeList;
    uint *nodePointer;
    uint *inDegree;
    uint *outDegree;
    long long numParts;
    PartPointer *partNodePointer;
    
	UMVirtualGraph(UMGraph &graph);
	void MakeGraph();
	void MakeUGraph();
};


#endif	//	UM_VIRTUAL_GRAPH_CUH
