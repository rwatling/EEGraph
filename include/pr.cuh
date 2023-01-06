#ifndef PR_CUH
#define PR_CUH 1

#include "globals.hpp"
#include "gpu_error_check.cuh"
#include "cuda_includes.cuh"
#include "virtual_graph.hpp"

namespace pr {
    __global__ void async_push_td(unsigned int numParts, 
								unsigned int *nodePointer, 
								PartPointer *partNodePointer,
								unsigned int *edgeList,
								float *dist,
								float *delta,
								bool* finished,
								float acc);

    __global__ void sync_push_dd(unsigned int numParts, 
								unsigned int *nodePointer, 
								PartPointer *partNodePointer,
								unsigned int *edgeList,
								float *dist,
								float *delta,
								bool* finished,
								float acc,
								bool* label1,
								bool* label2);

    __global__ void sync_push_td(unsigned int numParts, 
									unsigned int *nodePointer, 
									PartPointer *partNodePointer,
									unsigned int *edgeList,
									float *dist,
									float *delta,
									bool* finished,
									float acc,
									bool even);
    
    __global__ void async_push_dd(unsigned int numParts, 
								unsigned int *nodePointer, 
								PartPointer *partNodePointer,
								unsigned int *edgeList,
								float *dist,
								float *delta,
								bool* finished,
								float acc,
								bool* label1,
								bool* label2);

    bool checkSize(Graph graph, VirtualGraph vGraph, int deviceId);
}

#endif // PR_CUH