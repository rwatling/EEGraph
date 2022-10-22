#include "../include/sssp.cuh"

__global__ void sssp::async_push_td(  Edge* edges, 
                                      uint* weights, 
                                      uint num_edges,
                                      uint edges_per_thread, 
                                      int source,
                                      unsigned int* dist,
                                      bool* finished) {

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

    // Execute edge relaxation
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

__global__ void sssp::sync_push_td(  Edge* edges, 
                                     uint* weights, 
                                     uint num_edges,
                                     uint edges_per_thread, 
                                     int source,
                                     unsigned int* dist,
									 bool* finished,
									 bool evenPass  ) {

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

	// Execute edge relaxation
	if (evenPass) {

		for (int partId = startId; partId < endId; partId += 2) {
			e = edges[partId];
			e_w8 = weights[partId];
			final_dist = dist[e.source] + e_w8;

			if (final_dist < dist[e.end]) {
				atomicMin(&dist[e.end], final_dist);
				*finished = false;
			}
		}
	} else {
		for (int partId = startId + 1; partId < endId; partId += 2) {
			e = edges[partId];
			e_w8 = weights[partId];
			final_dist = dist[e.source] + e_w8;

			if (final_dist < dist[e.end]) {
				atomicMin(&dist[e.end], final_dist);
				*finished = false;
			}
		}
	}
}

__global__ void sssp::sync_push_dd(  Edge* edges, 
                                     uint* weights, 
                                     uint num_edges,
                                     uint edges_per_thread, 
                                     int source,
                                     unsigned int* dist,
									 bool* finished,
									 bool* active_current,
									 bool* active_next,
									 bool evenPass  ) {

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

	// Execute edge relaxation
	if (evenPass) {

		for (int partId = startId; partId < endId; partId += 2) {
			e = edges[partId];
			
			if (active_current[e.source]) {
				e_w8 = weights[partId];
				final_dist = dist[e.source] + e_w8;

				if (final_dist < dist[e.end]) {
					atomicMin(&dist[e.end], final_dist);
					*finished = false;
				}
			}

			active_next[e.end] = true;
		}
	} else {
		for (int partId = startId + 1; partId < endId; partId += 2) {
			e = edges[partId];
			
			if (active_current[e.source]) {
				e_w8 = weights[partId];
				final_dist = dist[e.source] + e_w8;

				if (final_dist < dist[e.end]) {
					atomicMin(&dist[e.end], final_dist);
					*finished = false;
				}
			}

			active_next[e.end] = true;
		}
	}
}

__global__ void sssp::sync_dd_clear_active(bool* active_list, 
											uint num_nodes,
											uint nodes_per_thread) {

	// Get identifiers
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int startId = threadId * nodes_per_thread;
    
    if (startId >= num_nodes) {
        return;
    }
    
    int endId = (threadId + 1) * nodes_per_thread;
    if (endId >= num_nodes) {
        endId = num_nodes;
    }
	for (int partId = startId; partId < endId; partId++) {
		active_list[partId] = false;
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
			e = edges[i];
			e_w8 = weights[i];
			final_dist = dist[e.source] + e_w8;

			if (final_dist < dist[e.end]) {
				dist[e.end] = final_dist;
				finished = false;
			}
		}
	}
}