#ifndef UM_VIRTUAL_GRAPH_CUH
#define UM_VIRTUAL_GRAPH_CUH

#include "um_graph.cuh"
#include "um_globals.cuh"
#include "globals.hpp"
#include "managed.cuh"

class UMVirtualGraph : public Managed
{
private:

public:
	UMGraph *graph;
	uint *edgeList;
    uint *nodePointer;
    uint *inDegree;
    uint *outDegree;
    long long numParts;
    UMPartPointer *partNodePointer;
    
	UMVirtualGraph(UMGraph &graph);
	void MakeGraph();
	void MakeUGraph();
};


#endif	//	UM_VIRTUAL_GRAPH_CUH
