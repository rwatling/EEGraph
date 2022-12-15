#ifndef BFS_CUH
#define BFS_CUH 1

#include "globals.hpp"
#include "gpu_error_check.cuh"
#include "cuda_includes.cuh"
#include "virtual_graph.hpp"
#include "um_virtual_graph.cuh"

namespace bfs {
    __global__ void async_push_dd(  unsigned int numParts, 
                                        unsigned int *nodePointer,
                                        PartPointer *partNodePointer, 
                                        unsigned int *edgeList,
                                        unsigned int* dist,
                                        bool* finished,
                                        bool* label1,
									    bool* label2);
    
    __global__ void async_push_td(  unsigned int numParts, 
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

    bool checkSize(Graph graph, VirtualGraph vGraph, int deviceId);
}

#endif //BFS_CUH