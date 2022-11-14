#ifndef UM_VIRTUAL_GRAPH_CUH
#define UM_VIRTUAL_GRAPH_CUH

#include "um_graph.cuh"
#include "um_globals.cuh"
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
    PartPointer *partNodePointer;
    
	UMVirtualGraph(UMGraph &graph);
	void MakeGraph();
	void MakeUGraph();
};


#endif	//	UM_VIRTUAL_GRAPH_CUH
