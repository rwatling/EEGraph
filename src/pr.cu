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

__global__ void pr::async_push_td(  unsigned int numParts, 
                                     unsigned int *nodePointer,
									 PartPointer *partNodePointer, 
                                     unsigned int *edgeList,
                                     unsigned int* dist,
									 bool* finished) 
{
   int partId = blockDim.x * blockIdx.x + threadIdx.x;

	if(partId < numParts)
	{
	}
}

__global__ void pr::sync_push_td(unsigned int numParts, 
								unsigned int *nodePointer, 
								PartPointer *partNodePointer,
								unsigned int *edgeList,
								float *pr1,
								float *pr2) 
{

	int partId = blockDim.x * blockIdx.x + threadIdx.x;

	if(partId < numParts)
	{
		int id = partNodePointer[partId].node;
		int part = partNodePointer[partId].part;
		
		int thisPointer = nodePointer[id];
		int degree = edgeList[thisPointer];
		
		float sourcePR = (float) pr2[id] / degree;
			
		int numParts;
		if(degree % Part_Size == 0)
			numParts = degree / Part_Size ;
		else
			numParts = degree / Part_Size + 1;
		
		int end;
		int ofs = thisPointer + part + 1;

		for(int i=0; i<Part_Size; i++)
		{
			if(part + i*numParts >= degree)
				break;
			end = ofs + i*numParts;

			atomicAdd(&pr1[edgeList[end]], sourcePR);
		}	
	}
}

__global__ void pr::sync_push_dd(unsigned int numParts, 
								unsigned int *nodePointer, 
								PartPointer *partNodePointer,
								unsigned int *edgeList,
								float *pr1,
								float *pr2,
								bool* label1,
								bool* label2) 
{

	int partId = blockDim.x * blockIdx.x + threadIdx.x;

	if(partId < numParts)
	{
		int id = partNodePointer[partId].node;
		int part = partNodePointer[partId].part;
		
		if(label1[id] == false) return;

		int thisPointer = nodePointer[id];
		int degree = edgeList[thisPointer];
		
		float sourcePR = (float) pr2[id] / degree;
			
		int numParts;
		if(degree % Part_Size == 0)
			numParts = degree / Part_Size ;
		else
			numParts = degree / Part_Size + 1;
		
		int end;
		int ofs = thisPointer + part + 1;

		for(int i=0; i<Part_Size; i++)
		{
			if(part + i*numParts >= degree)
				break;
			end = ofs + i*numParts;

			atomicAdd(&pr1[edgeList[end]], sourcePR);
			label2[edgeList[end]] = true;
		}	
	}
}

__global__ void pr::async_push_dd(  unsigned int numParts, 
                                     unsigned int *nodePointer,
									 PartPointer *partNodePointer, 
                                     unsigned int *edgeList,
                                     unsigned int* dist,
									 bool* finished,
									 bool* label1,
									 bool* label2) 
{
    
	int partId = blockDim.x * blockIdx.x + threadIdx.x;

	if(partId < numParts)
	{	
	}
}

void pr::seq_cpu(unsigned int numParts, 
				unsigned int *nodePointer, 
				PartPointer *partNodePointer,
				unsigned int *edgeList,
				float *pr1,
				float *pr2)
{

	for (int partId = 0; partId < numParts; partId++) {
		int id = partNodePointer[partId].node;
		int part = partNodePointer[partId].part;
		
		int thisPointer = nodePointer[id];
		int degree = edgeList[thisPointer];
		
		float sourcePR = (float) pr2[id] / degree;
			
		int numParts;
		if(degree % Part_Size == 0)
			numParts = degree / Part_Size ;
		else
			numParts = degree / Part_Size + 1;
		
		int end;
		int ofs = thisPointer + part + 1;

		for(int i=0; i<Part_Size; i++)
		{
			if(part + i*numParts >= degree)
				break;
			end = ofs + i*numParts;

			pr1[edgeList[end]] += sourcePR;
		}	
	}

}

__global__ void pr::clearLabel(bool *label, unsigned int size)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < size)
		label[id] = false;
}

__global__ void pr::clearVal(float *prA, float *prB, unsigned int num_nodes, float base)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < num_nodes)
	{
		prA[id] = base + prA[id] * 0.85;
		prB[id] = 0;
	}
}

void pr::cpu_clearVal(float *prA, float *prB, unsigned int num_nodes, float base)
{
	for (int i = 0; i < num_nodes; i++) {
		prA[i] = base + prA[i] * 0.85;
		prB[i] = 0;
	}
}