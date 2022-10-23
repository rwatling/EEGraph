#ifndef SSSP_CUH
#define SSSP_CUH 1

#include "globals.hpp"
#include "gpu_error_check.cuh"
#include "cuda_includes.cuh"
#include "virtual_graph.hpp"

namespace sssp {
    __global__ void async_push_td(  unsigned int numParts, 
                                   unsigned int *nodePointer,
                                   PartPointer *partNodePointer,
                                   unsigned int *edgeList,
                                   unsigned int* dist,
								   bool* finished);

    __global__ void sync_push_td(  Edge* edges, 
                                   uint* weights, 
                                   uint num_edges,
                                   uint edges_per_thread, 
                                   int source,
                                   unsigned int* dist,
                                   bool* finished,
                                   bool evenPass );

    __global__ void sync_push_dd(  unsigned int numParts, 
                                   unsigned int *nodePointer,
                                   PartPointer *partNodePointer,
                                   unsigned int *edgeList,
                                   unsigned int* dist,
								   bool* finished,
								   bool* label1,
								   bool* label2);

    void seq_cpu(  vector<Edge> edges, 
                   vector<uint> weights, 
                   uint num_edges, 
                   int source, 
                   unsigned int* dist  );
}

#endif // SSSP_CUH