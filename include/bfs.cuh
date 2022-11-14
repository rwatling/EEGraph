#ifndef BFS_CUH
#define BFS_CUH 1

#include "globals.hpp"
#include "gpu_error_check.cuh"
#include "cuda_includes.cuh"
#include "virtual_graph.hpp"

namespace bfs {
    __global__ void async_push_td( unsigned int numParts, 
                                   unsigned int *nodePointer,
                                   PartPointer *partNodePointer,
                                   unsigned int *edgeList,
                                   unsigned int* dist,
								   bool* finished,
                                   unsigned int level);

    __global__ void sync_push_dd(  unsigned int numParts, 
                                   unsigned int *nodePointer,
                                   PartPointer *partNodePointer,
                                   unsigned int *edgeList,
                                   unsigned int* dist,
								   bool* finished,
								   bool* label1,
								   bool* label2,
                                   unsigned int level);

    __global__ void sync_push_td(  unsigned int numParts, 
                                   unsigned int *nodePointer,
                                   PartPointer *partNodePointer, 
                                   unsigned int *edgeList,
                                   unsigned int* dist,
                                   bool* finished,
                                   bool even,
                                   unsigned int level);

    __global__ void async_push_dd(  unsigned int numParts, 
                                    unsigned int *nodePointer,
									PartPointer *partNodePointer, 
                                    unsigned int *edgeList,
                                    unsigned int* dist,
									bool* finished,
                                    bool* label1,
                                    bool* label2,
                                    unsigned int level);

    __global__ void clearLabel(bool *label, unsigned int size);

    bool checkSize(Graph graph, VirtualGraph vGraph, int deviceId);

    void seq_cpu(  vector<Edge> edges, 
                    vector<uint> weights, 
                    uint num_edges, 
                    int source, 
                    unsigned int* dist);
}

#endif //BFS_CUH