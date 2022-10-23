#include "../include/sssp.cuh"

__global__ void sssp::async_push_td(  unsigned int numParts, 
                                     unsigned int *nodePointer,
									 PartPointer *partNodePointer, 
                                     unsigned int *edgeList,
                                     unsigned int* dist,
									 bool* finished) {
   int partId = blockDim.x * blockIdx.x + threadIdx.x;

	if(partId < numParts)
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
	
	}
}

__global__ void sssp::sync_push_td(  unsigned int numParts, 
                                     unsigned int *nodePointer,
									 PartPointer *partNodePointer, 
                                     unsigned int *edgeList,
                                     unsigned int* dist,
									 bool* finished,
									 bool even) {
   int partId = blockDim.x * blockIdx.x + threadIdx.x;

	if((partId < numParts) && (partId % 2 == 0) && even)
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
	
	} else if (partId < numParts && (partId % 2 == 1)) {
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
									 bool* label1) {
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

				label1[edgeList[end]] = true;
			}
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

__global__ void sssp::clearLabel(bool *label, unsigned int size)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < size)
		label[id] = false;
}