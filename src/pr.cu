#include "../include/pr.cuh"

bool pr::checkSize(Graph graph, VirtualGraph vGraph, int deviceId) {
	cudaProfilerStart();
	cudaError_t error;
	cudaDeviceProp dev;
	int deviceID;
	cudaGetDevice(&deviceID);
	error = cudaGetDeviceProperties(&dev, deviceID);
	if(error != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
	cudaProfilerStop();

	size_t total_size = 0;
	total_size = ((3 * graph.num_nodes) + (2 * graph.num_edges)) * sizeof(unsigned int); //edges and nodes
	total_size = total_size + (sizeof(bool) * (2  + graph.num_nodes * 2)); //finished and labels
	total_size = total_size + (sizeof(PartPointer) * vGraph.numParts); //Part pointers

	return (total_size < dev.totalGlobalMem);
}

__global__ void pr::async_push_td(  unsigned int numParts, 
                                     unsigned int *nodePointer,
									 PartPointer *partNodePointer, 
                                     unsigned int *edgeList,
                                     unsigned int* dist,
									 bool* finished) {
   int partId = blockDim.x * blockIdx.x + threadIdx.x;

	if(partId < numParts)
	{
	}
}

__global__ void pr::sync_push_td(  unsigned int numParts, 
                                     unsigned int *nodePointer,
									 PartPointer *partNodePointer, 
                                     unsigned int *edgeList,
                                     unsigned int* dist,
									 bool* finished,
									 bool even) {
   int partId = blockDim.x * blockIdx.x + threadIdx.x;

	if((partId < numParts) && (partId % 2 == 0) && even)
	{
		
	} else if (partId < numParts && (partId % 2 == 1)) {

	}
}

__global__ void pr::sync_push_dd(  unsigned int numParts, 
                                     unsigned int *nodePointer,
									 PartPointer *partNodePointer, 
                                     unsigned int *edgeList,
                                     unsigned int* dist,
									 bool* finished,
									 bool* label1,
									 bool* label2) {
   int partId = blockDim.x * blockIdx.x + threadIdx.x;

	if(partId < numParts)
	{
	}
}

__global__ void pr::async_push_dd(  unsigned int numParts, 
                                     unsigned int *nodePointer,
									 PartPointer *partNodePointer, 
                                     unsigned int *edgeList,
                                     unsigned int* dist,
									 bool* finished,
									 bool* label1,
									 bool* label2) {
    
	int partId = blockDim.x * blockIdx.x + threadIdx.x;

	if(partId < numParts)
	{	
	}
}

void pr::seq_cpu(  vector<Edge> edges, 
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

__global__ void pr::clearLabel(bool *label, unsigned int size)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < size)
		label[id] = false;
}