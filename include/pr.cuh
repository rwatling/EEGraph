#ifndef PR_CUH
#define PR_CUH 1

#include "globals.hpp"
#include "gpu_error_check.cuh"
#include "cuda_includes.cuh"
#include "virtual_graph.hpp"

namespace pr {
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

    __global__ void sync_push_td(unsigned int numParts, 
                                    unsigned int *nodePointer, 
                                    PartPointer *partNodePointer,
                                    unsigned int *edgeList,
                                    float *pr1,
                                    float *pr2);

    __global__ void async_push_dd(  unsigned int numParts, 
                                    unsigned int *nodePointer,
									PartPointer *partNodePointer, 
                                    unsigned int *edgeList,
                                    unsigned int* dist,
									bool* finished,
                                    bool* label1,
                                    bool* label2);

    __global__ void clearLabel(bool *label, unsigned int size);

    __global__ void clearVal(float *prA, float *prB, unsigned int num_nodes, float base);

    bool checkSize(Graph graph, VirtualGraph vGraph, int deviceId);

    void seq_cpu(unsigned int numParts, 
								unsigned int *nodePointer, 
								PartPointer *partNodePointer,
								unsigned int *edgeList,
								float *pr1,
								float *pr2);
    
    void cpu_clearVal(float *prA, float *prB, unsigned int num_nodes, float base);
}

#endif // PR_CUH