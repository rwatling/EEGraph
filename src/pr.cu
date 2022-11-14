#include "../include/pr.cuh"

bool pr::checkSize(Graph graph, VirtualGraph vGraph, int deviceId) 
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

__global__ void pr::sync_push_td(unsigned int numParts, 
								unsigned int *nodePointer, 
								PartPointer *partNodePointer,
								unsigned int *edgeList,
								float *dist,
								float *delta,
								bool* finished,
								float acc,
								bool even) 
{

	int partId = blockDim.x * blockIdx.x + threadIdx.x;

	if(partId < numParts && (partId % 2 == 0) && even)
	{
		int id = partNodePointer[partId].node;
		int part = partNodePointer[partId].part;

		int thisPointer = nodePointer[id];
		int degree = edgeList[thisPointer];

		float thisDelta = delta[id];		

		if (thisDelta > acc) {

			dist[id] += thisDelta;

			if (degree != 0) {

				*finished = false;

				int numParts;
				if(degree % Part_Size == 0)
					numParts = degree / Part_Size ;
				else
					numParts = degree / Part_Size + 1;

				int end;
				int ofs = thisPointer + part + 1;

				float sourcePR = ((float) thisDelta / degree) * 0.85;

				for(int i=0; i<Part_Size; i++)
				{
					if(part + i*numParts >= degree)
						break;
					end = ofs + i*numParts;

					atomicAdd(&delta[edgeList[end]], sourcePR);
				}
			}

			atomicAdd(&delta[id], -thisDelta);
		}
	} else if (partId < numParts && (partId % 2 == 1)) {
		int id = partNodePointer[partId].node;
		int part = partNodePointer[partId].part;

		int thisPointer = nodePointer[id];
		int degree = edgeList[thisPointer];

		float thisDelta = delta[id];		

		if (thisDelta > acc) {

			dist[id] += thisDelta;

			if (degree != 0) {

				*finished = false;

				int numParts;
				if(degree % Part_Size == 0)
					numParts = degree / Part_Size ;
				else
					numParts = degree / Part_Size + 1;

				int end;
				int ofs = thisPointer + part + 1;

				float sourcePR = ((float) thisDelta / degree) * 0.85;

				for(int i=0; i<Part_Size; i++)
				{
					if(part + i*numParts >= degree)
						break;
					end = ofs + i*numParts;

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
	int partId = blockDim.x * blockIdx.x + threadIdx.x;

	if(partId < numParts)
	{
		int id = partNodePointer[partId].node;
		int part = partNodePointer[partId].part;

		int thisPointer = nodePointer[id];
		int degree = edgeList[thisPointer];

		if(label1[id] == false)
			return;

		float thisDelta = delta[id];		

		if (thisDelta > acc) {

			dist[id] += thisDelta;

			if (degree != 0) {

				*finished = false;

				int numParts;
				if(degree % Part_Size == 0)
					numParts = degree / Part_Size ;
				else
					numParts = degree / Part_Size + 1;

				int end;
				int ofs = thisPointer + part + 1;

				float sourcePR = ((float) thisDelta / degree) * 0.85;

				for(int i=0; i<Part_Size; i++)
				{
					if(part + i*numParts >= degree)
						break;
					end = ofs + i*numParts;

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
								float acc) 
{
	int partId = blockDim.x * blockIdx.x + threadIdx.x;

	if(partId < numParts)
	{
		int id = partNodePointer[partId].node;
		int part = partNodePointer[partId].part;

		int thisPointer = nodePointer[id];
		int degree = edgeList[thisPointer];

		float thisDelta = delta[id];		

		if (thisDelta > acc) {

			dist[id] += thisDelta;

			if (degree != 0) {

				*finished = false;

				int numParts;
				if(degree % Part_Size == 0)
					numParts = degree / Part_Size ;
				else
					numParts = degree / Part_Size + 1;

				int end;
				int ofs = thisPointer + part + 1;

				float sourcePR = ((float) thisDelta / degree) * 0.85;

				for(int i=0; i<Part_Size; i++)
				{
					if(part + i*numParts >= degree)
						break;
					end = ofs + i*numParts;

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
	int partId = blockDim.x * blockIdx.x + threadIdx.x;

	if(partId < numParts)
	{
		int id = partNodePointer[partId].node;
		int part = partNodePointer[partId].part;

		int thisPointer = nodePointer[id];
		int degree = edgeList[thisPointer];

		if(label1[id] == false)
			return;

		label1[id] = false;

		float thisDelta = delta[id];		

		if (thisDelta > acc) {

			dist[id] += thisDelta;

			if (degree != 0) {

				*finished = false;

				int numParts;
				if(degree % Part_Size == 0)
					numParts = degree / Part_Size ;
				else
					numParts = degree / Part_Size + 1;

				int end;
				int ofs = thisPointer + part + 1;

				float sourcePR = ((float) thisDelta / degree) * 0.85;

				for(int i=0; i<Part_Size; i++)
				{
					if(part + i*numParts >= degree)
						break;
					end = ofs + i*numParts;

					atomicAdd(&delta[edgeList[end]], sourcePR);
					label2[edgeList[end]] = true;
				}
			}

			atomicAdd(&delta[id], -thisDelta);
		}
	}
}