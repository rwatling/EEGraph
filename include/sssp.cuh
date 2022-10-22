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

    __global__ void sync_push_td(  Edge* edges, 
                                   uint* weights, 
                                   uint num_edges,
                                   uint edges_per_thread, 
                                   int source,
                                   unsigned int* dist,
                                   bool* finished,
                                   bool evenPass );

    __global__ void sync_push_dd(  Edge* edges, 
                                   uint* weights, 
                                   uint num_edges,
                                   uint edges_per_thread, 
                                   int source,
                                   unsigned int* dist,
                                   bool* finished,
                                   bool* active_current,
                                   bool* active_next,
                                   bool evenPass );

    __global__ void sync_dd_clear_active(bool* active_list,
                                        uint num_nodes,
                                        uint nodes_per_thread);

    void seq_cpu(  vector<Edge> edges, 
                   vector<uint> weights, 
                   uint num_edges, 
                   int source, 
                   unsigned int* dist  );
}

#endif // SSSP_CUH