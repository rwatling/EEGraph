#ifndef SSWP_CUH
#define SSWP_CUH 1

#include "globals.hpp"
#include "gpu_error_check.cuh"
#include "cuda_includes.cuh"
#include "virtual_graph.hpp"

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
                                   bool even);

    __global__ void async_push_dd(  unsigned int numParts, 
                                    unsigned int *nodePointer,
									PartPointer *partNodePointer, 
                                    unsigned int *edgeList,
                                    unsigned int* dist,
									bool* finished,
                                    bool* label1,
                                    bool* label2);

    __global__ void clearLabel(bool *label, unsigned int size);

    bool checkSize(Graph graph, VirtualGraph vGraph, int deviceId);

    void seq_cpu(  vector<Edge> edges, 
                    vector<uint> weights, 
                    uint num_edges, 
                    int source, 
                    unsigned int* dist  );
}

#endif // SSWP_CUH