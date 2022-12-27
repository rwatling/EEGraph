#ifndef SSWP_CUH
#define SSWP_CUH 1

#include "globals.hpp"
#include "gpu_error_check.cuh"
#include "cuda_includes.cuh"
#include "virtual_graph.hpp"
#include "um_virtual_graph.cuh"

namespace sswp {
    __global__ void async_push_td( unsigned int numParts, 
                                   unsigned int *nodePointer,
                                   PartPointer *partNodePointer,
                                   unsigned int *edgeList,
                                   unsigned int* dist,
								   bool* finished);

    __global__ void sync_push_dd(  unsigned int numParts, 
                                   unsigned int *nodePointer,
                                   PartPointer *partNodePointer,
                                   unsigned int *edgeList,
                                   unsigned int* dist,
								   bool* finished,
								   bool* label1,
								   bool* label2);

    __global__ void sync_push_td(  unsigned int numParts, 
                                   unsigned int *nodePointer,
                                   PartPointer *partNodePointer, 
                                   unsigned int *edgeList,
                                   unsigned int* dist,
                                   bool* finished,
                                   bool odd);

    __global__ void async_push_dd(  unsigned int numParts, 
                                    unsigned int *nodePointer,
									PartPointer *partNodePointer, 
                                    unsigned int *edgeList,
                                    unsigned int* dist,
									bool* finished,
                                    bool* label1,
                                    bool* label2);

    bool checkSize(Graph graph, VirtualGraph vGraph, int deviceId);

    void seq_cpu(  vector<Edge> edges, 
                    vector<uint> weights, 
                    uint num_edges,  
                    unsigned int* dist  );
    
    void seq_cpu(VirtualGraph vGraph, unsigned int* dist);

    void seq_cpu(UMVirtualGraph vGraph, unsigned int* dist);
}

#endif // SSSP_CUH