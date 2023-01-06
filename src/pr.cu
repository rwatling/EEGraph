#include "../include/pr.cuh"

__global__ void pr::sync_push_td(unsigned int numParts, 
								unsigned int *nodePointer, 
								PartPointer *partNodePointer,
								unsigned int *edgeList,
								float *dist,
								float *delta,
								bool* finished,
								float acc,
								bool odd) 
{

	unsigned int partId = blockDim.x * blockIdx.x + threadIdx.x;

	if((partId < numParts) && odd)
	{
		unsigned int id = partNodePointer[partId].node;
		unsigned int part = partNodePointer[partId].part;

		unsigned int thisPointer = nodePointer[id];
		unsigned int degree = edgeList[thisPointer];

		float thisDelta = delta[id];

		unsigned int numParts;

		if (thisDelta > acc) {

			dist[id] += thisDelta;

			if (degree != 0) {
				*finished = false;

				if(degree % Part_Size == 0)
					numParts = degree / Part_Size ;
				else
					numParts = degree / Part_Size + 1;
				
				unsigned int end;
				unsigned int ofs = thisPointer + 2*part +1;

				float sourcePR = ((float) thisDelta / degree) * 0.85;

				for(unsigned int i=0; i<Part_Size; i++)
				{
					if(part + i*numParts >= degree)
						break;
					end = ofs + i*numParts*2;
					
					atomicAdd(&delta[edgeList[end]], sourcePR);
				}
			}

			atomicAdd(&delta[id], -thisDelta);
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

		float thisDelta = delta[id];

		unsigned int thisPointer = nodePointer[id];
		unsigned int degree = edgeList[thisPointer];

		unsigned int numParts;

		if (thisDelta > acc) {
			dist[id] += thisDelta;

			if (degree != 0) {
				*finished = false;

				if(degree % Part_Size == 0)
					numParts = degree / Part_Size ;
				else
					numParts = degree / Part_Size + 1;
				
				unsigned int end;
				unsigned int ofs = thisPointer + 2*part +1;

				float sourcePR = ((float) thisDelta / degree) * 0.85;

				for(unsigned int i=0; i<Part_Size; i++)
				{
					if(part + i*numParts >= degree)
						break;
					end = ofs + i*numParts*2;
					
					atomicAdd(&delta[edgeList[end]], sourcePR);
				}
			}

			atomicAdd(&delta[id], -thisDelta);
		}
	}
}

__global__ void pr::sync_push_dd(unsigned int numParts, 
								unsigned int *nodePointer, 
								PartPointer *partNodePointer,
								unsigned int *edgeList,
								float *dist,
								float *delta,
								bool* finished,
								float acc,
								bool* label1,
								bool* label2) 
{

	unsigned int partId = blockDim.x * blockIdx.x + threadIdx.x;

	if (partId < numParts) {
		unsigned int id = partNodePointer[partId].node;
		unsigned int part = partNodePointer[partId].part;

		if(label1[id] == false)
			return;

		unsigned int thisPointer = nodePointer[id];
		unsigned int degree = edgeList[thisPointer];

		float thisDelta = delta[id];

		unsigned int numParts;

		if (thisDelta > acc) {

			dist[id] += thisDelta;

			if (degree != 0) {
				*finished = false;

				if(degree % Part_Size == 0)
					numParts = degree / Part_Size ;
				else
					numParts = degree / Part_Size + 1;
				
				unsigned int end;
				unsigned int ofs = thisPointer + 2*part +1;

				float sourcePR = ((float) thisDelta / degree) * 0.85;

				for(unsigned int i=0; i<Part_Size; i++)
				{
					if(part + i*numParts >= degree)
						break;
					end = ofs + i*numParts*2;
					
					atomicAdd(&delta[edgeList[end]], sourcePR);
					label2[edgeList[end]] = true;
				}
			}

			atomicAdd(&delta[id], -thisDelta);
		}
	}
}

__global__ void pr::async_push_td(unsigned int numParts, 
								unsigned int *nodePointer, 
								PartPointer *partNodePointer,
								unsigned int *edgeList,
								float *dist,
								float *delta,
								bool* finished,
								float acc) {
	
	unsigned int partId = blockDim.x * blockIdx.x + threadIdx.x;

	if(partId < numParts) {
		unsigned int id = partNodePointer[partId].node;
		unsigned int part = partNodePointer[partId].part;

		unsigned int thisPointer = nodePointer[id];
		unsigned int degree = edgeList[thisPointer];

		float thisDelta = delta[id];

		unsigned int numParts;

		if (thisDelta > acc) {

			dist[id] += thisDelta;

			if (degree != 0) {
				*finished = false;

				if(degree % Part_Size == 0)
					numParts = degree / Part_Size ;
				else
					numParts = degree / Part_Size + 1;
				
				unsigned int end;
				unsigned int ofs = thisPointer + 2*part +1;

				float sourcePR = ((float) thisDelta / degree) * 0.85;

				for(unsigned int i=0; i<Part_Size; i++)
				{
					if(part + i*numParts >= degree)
						break;
					end = ofs + i*numParts*2;
					
					atomicAdd(&delta[edgeList[end]], sourcePR);
				}
			}

			atomicAdd(&delta[id], -thisDelta);
		}
	
	}

}

__global__ void pr::async_push_dd(unsigned int numParts, 
								unsigned int *nodePointer, 
								PartPointer *partNodePointer,
								unsigned int *edgeList,
								float *dist,
								float *delta,
								bool* finished,
								float acc,
								bool* label1,
								bool* label2) 
{
	unsigned int partId = blockDim.x * blockIdx.x + threadIdx.x;

	if(partId < numParts) {
		unsigned int id = partNodePointer[partId].node;
		unsigned int part = partNodePointer[partId].part;

		unsigned int thisPointer = nodePointer[id];
		unsigned int degree = edgeList[thisPointer];

		float thisDelta = delta[id];

		unsigned int numParts;

		if(label1[id] == false)
			return;

		label1[id] = false;

		if (thisDelta > acc) {

			dist[id] += thisDelta;

			if (degree != 0) {
				*finished = false;

				if(degree % Part_Size == 0)
					numParts = degree / Part_Size ;
				else
					numParts = degree / Part_Size + 1;
				
				unsigned int end;
				unsigned int ofs = thisPointer + 2*part +1;

				float sourcePR = ((float) thisDelta / degree) * 0.85;

				for(unsigned int i=0; i<Part_Size; i++)
				{
					if(part + i*numParts >= degree)
						break;
					end = ofs + i*numParts*2;
					
					atomicAdd(&delta[edgeList[end]], sourcePR);
					label2[edgeList[end]] = true;
				}
			}

			atomicAdd(&delta[id], -thisDelta);
		}
	
	}
}