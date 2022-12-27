#include "../include/sswp.cuh"

bool sswp::checkSize(Graph graph, VirtualGraph vGraph, int deviceId) 
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

__global__ void sswp::async_push_td(  unsigned int numParts, 
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
			finalDist = min(sourceWeight, edgeList[w8]);
			if(finalDist > dist[edgeList[end]])
			{
				atomicMax(&dist[edgeList[end]] , finalDist);
				*finished = false;
			}
		}
	
	}
}

__global__ void sswp::sync_push_td(  unsigned int numParts, 
                                     unsigned int *nodePointer,
									 PartPointer *partNodePointer, 
                                     unsigned int *edgeList,
                                     unsigned int* dist,
									 bool* finished,
									 bool odd) {
   unsigned int partId = blockDim.x * blockIdx.x + threadIdx.x;

	if((partId < numParts) && odd)
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

		for(unsigned int i=0; i<Part_Size; i++)
		{
			if(part + i*numParts >= degree)
				break;
			end = ofs + i*numParts*2;
			w8 = end + 1;
			finalDist = min(sourceWeight, edgeList[w8]);
			if(finalDist > dist[edgeList[end]])
			{
				atomicMax(&dist[edgeList[end]] , finalDist);
				*finished = false;
			}
		}
	
	} else if (partId < (numParts + 1)) {
		
		// Each thread swaps with a partner
		if (partId % 2 == 0) {
			partId = partId + 1;
		} else if (partId % 2 == 1) {
			partId = partId - 1;
		}

		if (partId >= numParts) return;

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

		for(unsigned int i=0; i<Part_Size; i++)
		{
			if(part + i*numParts >= degree)
				break;
			end = ofs + i*numParts*2;
			w8 = end + 1;
			finalDist = min(sourceWeight, edgeList[w8]);
			if(finalDist > dist[edgeList[end]])
			{
				atomicMax(&dist[edgeList[end]] , finalDist);
				*finished = false;
			}
		}
	
	}
}

__global__ void sswp::sync_push_dd(  unsigned int numParts, 
                                     unsigned int *nodePointer,
									 PartPointer *partNodePointer, 
                                     unsigned int *edgeList,
                                     unsigned int* dist,
									 bool* finished,
									 bool* label1,
									 bool* label2) {
   unsigned int partId = blockDim.x * blockIdx.x + threadIdx.x;

	if(partId < numParts)
	{
		unsigned int id = partNodePointer[partId].node;
		unsigned int part = partNodePointer[partId].part;

		if(label1[id] == false)
			return;

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

		for(unsigned int i=0; i<Part_Size; i++)
		{
			if(part + i*numParts >= degree)
				break;
			end = ofs + i*numParts*2;
			w8 = end + 1;
			finalDist = min(sourceWeight, edgeList[w8]);
			if(finalDist > dist[edgeList[end]])
			{
				atomicMax(&dist[edgeList[end]] , finalDist);
				*finished = false;

				label2[edgeList[end]] = true;
			}
		}
	
	}
}

__global__ void sswp::async_push_dd(  unsigned int numParts, 
                                     unsigned int *nodePointer,
									 PartPointer *partNodePointer, 
                                     unsigned int *edgeList,
                                     unsigned int* dist,
									 bool* finished,
									 bool* label1,
									 bool* label2) {
    
	unsigned int partId = blockDim.x * blockIdx.x + threadIdx.x;

	if(partId < numParts)
	{
		unsigned int id = partNodePointer[partId].node;
		unsigned int part = partNodePointer[partId].part;

		if(label1[id] == false)
			return;

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

		for(unsigned int i=0; i<Part_Size; i++)
		{
			if(part + i*numParts >= degree)
				break;
			end = ofs + i*numParts*2;
			w8 = end + 1;
			finalDist = min(sourceWeight, edgeList[w8]);
			if(finalDist > dist[edgeList[end]])
			{
				atomicMax(&dist[edgeList[end]] , finalDist);
				*finished = false;

				label2[edgeList[end]] = true;
			}
		}

		label1[id] = false;
	
	}
}

void sswp::seq_cpu(VirtualGraph vGraph, unsigned int* dist) {

	unsigned int numParts = vGraph.numParts;
	PartPointer* partNodePointer = vGraph.partNodePointer;
	unsigned int* edgeList = vGraph.edgeList;
	unsigned int* nodePointer = vGraph.nodePointer;

	bool finished;

	do {
		finished = true;

		for (unsigned int partId = 0; partId < numParts; partId++) {
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

			for(unsigned int i=0; i<Part_Size; i++)
			{
				if(part + i*numParts >= degree)
					break;
				end = ofs + i*numParts*2;
				w8 = end + 1;
				finalDist = min(sourceWeight, edgeList[w8]);
				if(finalDist > dist[edgeList[end]])
				{
					dist[edgeList[end]] = finalDist;
					finished = false;
				}
			}
		}
	} while (!(finished));
}

void sswp::seq_cpu(UMVirtualGraph vGraph, unsigned int* dist) {

	unsigned int numParts = vGraph.numParts;
	PartPointer* partNodePointer = vGraph.partNodePointer;
	unsigned int* edgeList = vGraph.edgeList;
	unsigned int* nodePointer = vGraph.nodePointer;

	bool finished = false;

	do {
		finished = true;

		for (unsigned int partId = 0; partId < numParts; partId++) {
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

			for(unsigned int i=0; i<Part_Size; i++)
			{
				if(part + i*numParts >= degree)
					break;
				end = ofs + i*numParts*2;
				w8 = end + 1;
				finalDist = min(sourceWeight, edgeList[w8]);
				if(finalDist > dist[edgeList[end]])
				{
					dist[edgeList[end]] = finalDist;
					finished = false;
				}
			}
		}
	} while (!finished);
}
