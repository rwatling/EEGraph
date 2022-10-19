#include "../include/sssp.cuh"

__global__ void sssp::async_push_td(  Edge* edges, 
                                      uint* weights, 
                                      uint num_edges,
                                      uint edges_per_thread, 
                                      int source,
                                      unsigned int* dist,
                                      bool* finished  ) {

    // Get identifiers
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int startId = threadId * edges_per_thread;
    
    if (startId >= num_edges) {
        return;
    }
    
    int endId = (threadId + 1) * edges_per_thread;
    if (endId >= num_edges) {
        endId = num_edges;
    }

	Edge e;
	uint e_w8;
	uint final_dist;

    // Execute relaxation
	for (int partId = startId; partId < endId; partId++) {
		e = edges[partId];
		e_w8 = weights[partId];
		final_dist = dist[e.source] + e_w8;

		if (final_dist < dist[e.end]) {
			atomicMin(&dist[e.end], final_dist);
			*finished = false;
		}
	}
}

void sssp::seq_cpu(  vector<Edge> edges, 
                     vector<uint> weights, 
                     uint num_edges, 
                     int source, 
                     unsigned int* dist  ) {

	bool finished = false;

	while (!finished) {
		finished = true;

		Edge e;
		uint e_w8;
		uint final_dist;

		for (int i = 0; i < num_edges; i++) {
			e = edges.at(i);
			e_w8 = weights.at(i);
			final_dist = dist[e.source] + e_w8;

			if (final_dist < dist[e.end]) {
				dist[e.end] = final_dist;
				finished = false;
			}
		}
	}
}