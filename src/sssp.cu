#include "../include/sssp.cuh"

bool sssp::checkSize(Graph graph, VirtualGraph vGraph, int deviceId) 
{
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

__global__ void sssp::async_push_td(  unsigned int numParts, 
                                     unsigned int *nodePointer,
									 PartPointer *partNodePointer, 
                                     unsigned int *edgeList,
                                     unsigned int* dist,
									 bool* finished)
{
   unsigned int partId = blockDim.x * blockIdx.x + threadIdx.x;

	if(partId < numParts)
	{
		unsigned int id = partNodePointer[partId].node;
		unsigned int part = partNodePointer[partId].part;

		unsigned int sourceWeight = dist[id];

		unsigned int thisPointer = nodePointer[id];
		unsigned int degree = edgeList[thisPointer];

		unsigned int numParts;
		if(degree % Part_Size == 0)
			numParts = degree / Part_Size ;
		else
			numParts = degree / Part_Size + 1;
		
		unsigned int end;
		unsigned int w8;
		unsigned int finalDist;
		unsigned int ofs = thisPointer + 2*part +1;

		for(int i=0; i<Part_Size; i++)
		{
			if(part + i*numParts >= degree)
				break;
			end = ofs + i*numParts*2;
			w8 = end + 1;
			finalDist = sourceWeight + edgeList[w8];
			if(finalDist < dist[edgeList[end]])
			{
				atomicMin(&dist[edgeList[end]] , finalDist);
				*finished = false;
			}
		}
	
	}
}

// Needs to be fixed
__global__ void sssp::sync_push_td(  unsigned int numParts, 
                                     unsigned int *nodePointer,
									 PartPointer *partNodePointer, 
                                     unsigned int *edgeList,
                                     unsigned int* dist,
									 bool* finished,
									 bool even) {
   int partId = blockDim.x * blockIdx.x + threadIdx.x;

	if((partId < numParts) && even)
	{
		int id = partNodePointer[partId].node;
		int part = partNodePointer[partId].part;

		int sourceWeight = dist[id];

		int thisPointer = nodePointer[id];
		int degree = edgeList[thisPointer];

		int numParts;
		if(degree % Part_Size == 0)
			numParts = degree / Part_Size ;
		else
			numParts = degree / Part_Size + 1;
		
		int end;
		int w8;
		int finalDist;
		int ofs = thisPointer + 2*part +1;

		for(int i=0; i<Part_Size; i++)
		{
			if(part + i*numParts >= degree)
				break;
			end = ofs + i*numParts*2;
			w8 = end + 1;
			finalDist = sourceWeight + edgeList[w8];
			if(finalDist < dist[edgeList[end]])
			{
				atomicMin(&dist[edgeList[end]] , finalDist);
				*finished = false;
			}
		}
	
	} else if (partId < numParts) {
		int id = partNodePointer[partId].node;
		int part = partNodePointer[partId].part;

		int sourceWeight = dist[id];

		int thisPointer = nodePointer[id];
		int degree = edgeList[thisPointer];

		int numParts;
		if(degree % Part_Size == 0)
			numParts = degree / Part_Size ;
		else
			numParts = degree / Part_Size + 1;
		
		int end;
		int w8;
		int finalDist;
		int ofs = thisPointer + 2*part +1;

		for(int i=0; i<Part_Size; i++)
		{
			if(part + i*numParts >= degree)
				break;
			end = ofs + i*numParts*2;
			w8 = end + 1;
			finalDist = sourceWeight + edgeList[w8];
			if(finalDist < dist[edgeList[end]])
			{
				atomicMin(&dist[edgeList[end]] , finalDist);
				*finished = false;
			}
		}
	
	}
}
__global__ void sssp::sync_push_dd(  unsigned int numParts, 
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
		int id = partNodePointer[partId].node;
		int part = partNodePointer[partId].part;

		if(label1[id] == false)
			return;

		int sourceWeight = dist[id];

		int thisPointer = nodePointer[id];
		int degree = edgeList[thisPointer];

		int numParts;
		if(degree % Part_Size == 0)
			numParts = degree / Part_Size ;
		else
			numParts = degree / Part_Size + 1;
		
		int end;
		int w8;
		int finalDist;
		int ofs = thisPointer + 2*part +1;

		for(int i=0; i<Part_Size; i++)
		{
			if(part + i*numParts >= degree)
				break;
			end = ofs + i*numParts*2;
			w8 = end + 1;
			finalDist = sourceWeight + edgeList[w8];
			if(finalDist < dist[edgeList[end]])
			{
				atomicMin(&dist[edgeList[end]] , finalDist);
				*finished = false;

				label2[edgeList[end]] = true;
			}
		}
	
	}
}

__global__ void sssp::async_push_dd(  unsigned int numParts, 
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
		int id = partNodePointer[partId].node;
		int part = partNodePointer[partId].part;

		if(label1[id] == false)
			return;

		label1[id] == false;

		int sourceWeight = dist[id];

		int thisPointer = nodePointer[id];
		int degree = edgeList[thisPointer];

		int numParts;
		if(degree % Part_Size == 0)
			numParts = degree / Part_Size ;
		else
			numParts = degree / Part_Size + 1;
		
		int end;
		int w8;
		int finalDist;
		int ofs = thisPointer + 2*part +1;

		for(int i=0; i<Part_Size; i++)
		{
			if(part + i*numParts >= degree)
				break;
			end = ofs + i*numParts*2;
			w8 = end + 1;
			finalDist = sourceWeight + edgeList[w8];
			if(finalDist < dist[edgeList[end]])
			{
				atomicMin(&dist[edgeList[end]] , finalDist);
				*finished = false;

				label2[edgeList[end]] = true;
			}
		}
	
	}
}

void sssp::seq_cpu(  vector<Edge> edges, 
                     vector<uint> weights, 
                     uint num_edges, 
                     unsigned int* dist  ) {

	bool finished = false;

	do {
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
	} while (!(finished));
}

/*void sssp::seq_cpu(  Edge* edges, 
                     uint* weights, 
                     uint num_edges, 
                     int source, 
                     unsigned int* dist  )
{

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
}*/

void sssp::seq_cpu(VirtualGraph vGraph, unsigned int* dist) {

	unsigned int numParts = vGraph.numParts;
	PartPointer* partNodePointer = vGraph.partNodePointer;
	unsigned int* edgeList = vGraph.edgeList;
	unsigned int* nodePointer = vGraph.nodePointer;

	bool finished;

	do {
		finished = true;

		for (unsigned int partId = 0; partId < numParts; partId++) {
			int id = partNodePointer[partId].node;
			int part = partNodePointer[partId].part;

			int sourceWeight = dist[id];

			int thisPointer = nodePointer[id];
			int degree = edgeList[thisPointer];

			int numParts;
			if(degree % Part_Size == 0)
				numParts = degree / Part_Size ;
			else
				numParts = degree / Part_Size + 1;
			
			int end;
			int w8;
			int finalDist;
			int ofs = thisPointer + 2*part +1;

			for(int i=0; i<Part_Size; i++)
			{
				if(part + i*numParts >= degree)
					break;
				end = ofs + i*numParts*2;
				w8 = end + 1;
				finalDist = sourceWeight + edgeList[w8];
				if(finalDist < dist[edgeList[end]])
				{
					dist[edgeList[end]] = min(dist[edgeList[end]], finalDist);
					finished = false;
				}
			}
		}
	} while (!(finished));
}

void sssp::seq_cpu(UMVirtualGraph vGraph, unsigned int* dist) {

	unsigned int numParts = vGraph.numParts;
	PartPointer* partNodePointer = vGraph.partNodePointer;
	unsigned int* edgeList = vGraph.edgeList;
	unsigned int* nodePointer = vGraph.nodePointer;

	bool finished = false;

	do {
		finished = true;

		for (int partId = 0; partId < numParts; partId++) {
			int id = partNodePointer[partId].node;
			int part = partNodePointer[partId].part;

			int sourceWeight = dist[id];

			int thisPointer = nodePointer[id];
			int degree = edgeList[thisPointer];

			int numParts;
			if(degree % Part_Size == 0)
				numParts = degree / Part_Size ;
			else
				numParts = degree / Part_Size + 1;
			
			int end;
			int w8;
			int finalDist;
			int ofs = thisPointer + 2*part +1;

			for(int i=0; i<Part_Size; i++)
			{
				if(part + i*numParts >= degree)
					break;
				end = ofs + i*numParts*2;
				w8 = end + 1;
				finalDist = sourceWeight + edgeList[w8];
				if(finalDist < dist[edgeList[end]])
				{
					dist[edgeList[end]] = finalDist;
					finished = false;
				}
			}
		}
	} while (!finished);
}
