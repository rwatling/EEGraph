#ifndef SSSP_CUH
#define SSSP_CUH 1

#include "globals.hpp"
#include "gpu_error_check.cuh"
#include "cuda_includes.cuh"

namespace sssp {
    __global__ void async_push_td(  Edge* edges, 
                                          uint* weights, 
                                          uint num_edges,
                                          uint edges_per_thread, 
                                          int source,
                                          unsigned int* dist,
                                          bool* finished  );


    void seq_cpu(  vector<Edge> edges, 
                         vector<uint> weights, 
                         uint num_edges, 
                         int source, 
                         unsigned int* dist  );
}

#endif // SSSP_CUH