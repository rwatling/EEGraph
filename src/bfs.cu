#include "../include/bfs.cuh"

bool bfs::checkSize(Graph graph, VirtualGraph vGraph, int deviceId) 
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

__global__ void bfs::async_push_td(  unsigned int numParts, 
                                     unsigned int *nodePointer,
									 PartPointer *partNodePointer, 
                                     unsigned int *edgeList,
                                     unsigned int* dist,
									 bool* finished,
									 unsigned int level)
{
   	int partId = blockDim.x * blockIdx.x + threadIdx.x;

	if(partId < numParts)
	{
		int id = partNodePointer[partId].node;
		int part = partNodePointer[partId].part;

		int thisPointer = nodePointer[id];
		int degree = edgeList[thisPointer];

		int numParts;
		if(degree % Part_Size == 0)
			numParts = degree / Part_Size ;
		else
			numParts = degree / Part_Size + 1;
		
		int end;
		int ofs = thisPointer + 2*part +1;

		for(int i=0; i<Part_Size; i++)
		{
			if(part + i*numParts >= degree)
				break;
			end = ofs + i*numParts*2;

			if((level + 1) < dist[edgeList[end]])
			{
				atomicMin(&dist[edgeList[end]], level + 1);
				*finished = false;
			}
		}
	
	}
}

__global__ void bfs::sync_push_td(  unsigned int numParts, 
                                     unsigned int *nodePointer,
									 PartPointer *partNodePointer, 
                                     unsigned int *edgeList,
                                     unsigned int* dist,
									 bool* finished,
									 bool even,
									 unsigned int level) 
{
   int partId = blockDim.x * blockIdx.x + threadIdx.x;

	if((partId < numParts) && (partId % 2 == 0) && even)
	{
		int id = partNodePointer[partId].node;
		int part = partNodePointer[partId].part;

		int thisPointer = nodePointer[id];
		int degree = edgeList[thisPointer];

		int numParts;
		if(degree % Part_Size == 0)
			numParts = degree / Part_Size ;
		else
			numParts = degree / Part_Size + 1;
		
		int end;
		int ofs = thisPointer + 2*part +1;

		for(int i=0; i<Part_Size; i++)
		{
			if(part + i*numParts >= degree)
				break;
			end = ofs + i*numParts*2;

			if((level + 1) < dist[edgeList[end]])
			{
				atomicMin(&dist[edgeList[end]], level + 1);
				*finished = false;
			}
		}
	
	} else if (partId < numParts && (partId % 2 == 1)) {
		int id = partNodePointer[partId].node;
		int part = partNodePointer[partId].part;

		int thisPointer = nodePointer[id];
		int degree = edgeList[thisPointer];

		int numParts;
		if(degree % Part_Size == 0)
			numParts = degree / Part_Size ;
		else
			numParts = degree / Part_Size + 1;
		
		int end;
		int ofs = thisPointer + 2*part +1;

		for(int i=0; i<Part_Size; i++)
		{
			if(part + i*numParts >= degree)
				break;
			end = ofs + i*numParts*2;

			if((level+1) < dist[edgeList[end]])
			{
				atomicMin(&dist[edgeList[end]], level + 1);
				*finished = false;
			}
		}
	
	}
}

__global__ void bfs::sync_push_dd(  unsigned int numParts, 
                                     unsigned int *nodePointer,
									 PartPointer *partNodePointer, 
                                     unsigned int *edgeList,
                                     unsigned int* dist,
									 bool* finished,
									 bool* label1,
									 bool* label2,
									 unsigned int level) 
{
   int partId = blockDim.x * blockIdx.x + threadIdx.x;

	if(partId < numParts)
	{
		int id = partNodePointer[partId].node;
		int part = partNodePointer[partId].part;

		if(label1[id] == false)
			return;
			
		//label1[id] = false;

		int thisPointer = nodePointer[id];
		int degree = edgeList[thisPointer];

		int numParts;
		if(degree % Part_Size == 0)
			numParts = degree / Part_Size ;
		else
			numParts = degree / Part_Size + 1;
		
		int end;
		int ofs = thisPointer + 2*part +1;

		for(int i=0; i<Part_Size; i++)
		{
			if(part + i*numParts >= degree)
				break;
			end = ofs + i*numParts*2;

			if((level+1) < dist[edgeList[end]])
			{
				atomicMin(&dist[edgeList[end]], level + 1);
				*finished = false;

				label2[edgeList[end]] = true;
			}
		}
	
	}
}

__global__ void bfs::async_push_dd(  unsigned int numParts, 
                                     unsigned int *nodePointer,
									 PartPointer *partNodePointer, 
                                     unsigned int *edgeList,
                                     unsigned int* dist,
									 bool* finished,
									 bool* label1,
									 bool* label2,
									 unsigned int level)
{
    
	int partId = blockDim.x * blockIdx.x + threadIdx.x;

	if(partId < numParts)
	{
		int id = partNodePointer[partId].node;
		int part = partNodePointer[partId].part;

		if(label1[id] == false)
			return;

		//label1[id] = false;

		int thisPointer = nodePointer[id];
		int degree = edgeList[thisPointer];

		int numParts;
		if(degree % Part_Size == 0)
			numParts = degree / Part_Size ;
		else
			numParts = degree / Part_Size + 1;
		
		int end;
		int ofs = thisPointer + 2*part +1;

		for(int i=0; i<Part_Size; i++)
		{
			if(part + i*numParts >= degree)
				break;
			end = ofs + i*numParts*2;

			if((level+1) < dist[edgeList[end]])
			{
				atomicMin(&dist[edgeList[end]], level + 1);
				*finished = false;

				label2[edgeList[end]] = true;
			}
		}
	
	}
}

void bfs::seq_cpu(  vector<Edge> edges, 
                     vector<uint> weights, 
                     uint num_edges, 
                     int source, 
                     unsigned int* dist)
{

	bool finished = false;
	unsigned int level = 0;

	while (!finished) {
		finished = true;

		Edge e;

		for (int i = 0; i < num_edges; i++) {
			e = edges[i];

			if (dist[e.end] == DIST_INFINITY) {
				dist[e.end] = level + 1;
				finished = false;
			}
		}

		level++;
	}
}

void bfs::seq_cpu(  Edge* edges, 
                     uint* weights, 
                     uint num_edges, 
                     int source, 
                     unsigned int* dist)
{

	bool finished = false;
	unsigned int level = 0;

	while (!finished) {
		finished = true;

		Edge e;

		for (int i = 0; i < num_edges; i++) {
			e = edges[i];

			if (dist[e.end] == DIST_INFINITY) {
				dist[e.end] = level + 1;
				finished = false;
			}
		}

		level++;
	}
}