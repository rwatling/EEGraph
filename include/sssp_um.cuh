#ifndef SSSP_UM_CUH
#define SSSP_UM_CUH 1

#include "um_globals.cuh"
#include "gpu_error_check.cuh"
#include "cuda_includes.cuh"
#include "um_virtual_graph.cuh"

namespace sssp_um {
    __global__ void async_push_td( unsigned int numParts, 
                                   unsigned int *nodePointer,
                                   UMPartPointer *partNodePointer,
                                   unsigned int *edgeList,
                                   unsigned int* dist,
								   bool* finished);

    __global__ void sync_push_dd(  unsigned int numParts, 
                                   unsigned int *nodePointer,
                                   UMPartPointer *partNodePointer,
                                   unsigned int *edgeList,
                                   unsigned int* dist,
								   bool* finished,
								   bool* label1,
								   bool* label2);

    __global__ void sync_push_td(  unsigned int numParts, 
                                   unsigned int *nodePointer,
                                   UMPartPointer *partNodePointer, 
                                   unsigned int *edgeList,
                                   unsigned int* dist,
                                   bool* finished,
                                   bool even);

    __global__ void async_push_dd(  unsigned int numParts, 
                                    unsigned int *nodePointer,
									UMPartPointer *partNodePointer, 
                                    unsigned int *edgeList,
                                    unsigned int* dist,
									bool* finished,
                                    bool* label1,
                                    bool* label2);

    __global__ void clearLabel(bool *label, unsigned int size);
    
    void seq_cpu(  UMEdge* edges, 
                     uint* weights, 
                     uint num_edges, 
                     int source, 
                     unsigned int* dist  );
}

#endif // SSSP_UM_CUH