#include "../include/um_virtual_graph.cuh"

UMVirtualGraph::UMVirtualGraph(UMGraph &graph)
{
	if(graph.hasZeroID == false)
	{
		for(int i=0; i<graph.num_edges; i++)
		{
			graph.edges[i].source = graph.edges[i].source - 1;
			graph.edges[i].end = graph.edges[i].end - 1;
		}
	}
	
	this->graph = &graph;
	
	cudaMallocManaged(&inDegree, sizeof(uint) * graph.num_nodes);
	cudaMallocManaged(&outDegree, sizeof(uint) * graph.num_nodes);

	for(int i=0; i<graph.num_nodes; i++)
	{
		outDegree[i] = 0;
		inDegree[i] = 0;
	}
	
	for(int i=0; i<graph.num_edges; i++)
	{
		outDegree[graph.edges[i].source]++;
		inDegree[graph.edges[i].end]++;
	}
	
}
	
void UMVirtualGraph::MakeGraph()
{ 
	cudaMallocManaged(&nodePointer, sizeof(uint) * graph->num_nodes);
	cudaMallocManaged(&edgeList, sizeof(uint) * (2* graph->num_edges + graph->num_nodes));

	uint *outDegreeCounter;
	uint source;
	uint end;
	uint w8;		
	
	long long counter=0;
	numParts = 0;
	int numZero = 0;
	
	for(int i=0; i<graph->num_nodes; i++)
	{
		nodePointer[i] = counter;
		edgeList[counter] = outDegree[i];

		if(outDegree[i] == 0)
			numZero++;
		
		if(outDegree[i] % Part_Size == 0)
			numParts += outDegree[i] / Part_Size ;
		else
			numParts += outDegree[i] / Part_Size + 1;
		
		counter = counter + outDegree[i]*2 + 1;
	}

	cudaMallocManaged(&outDegreeCounter, sizeof(uint) * graph->num_nodes);

	for(int i=0; i<graph->num_edges; i++)
	{
		source = graph->edges[i].source;
		end = graph->edges[i].end;
		w8 = graph->weights[i];
		
		uint location = nodePointer[source]+1+2*outDegreeCounter[source];

		edgeList[location] = end;
		edgeList[location+1] = w8;

		outDegreeCounter[source]++;
	}
	
	cudaFree(outDegreeCounter);
	cudaMallocManaged(&partNodePointer, sizeof(PartPointer) * numParts);

	int thisNumParts;
	long long countParts = 0;
	for(int i=0; i<graph->num_nodes; i++)
	{
		if(outDegree[i] % Part_Size == 0)
			thisNumParts = outDegree[i] / Part_Size ;
		else
			thisNumParts = outDegree[i] / Part_Size + 1;
		for(int j=0; j<thisNumParts; j++)
		{
			partNodePointer[countParts].node = i;
			partNodePointer[countParts++].part = j;
		}
	}
}


void UMVirtualGraph::MakeUGraph()
{
	cudaMallocManaged(&nodePointer, sizeof(uint) * graph->num_nodes);
	cudaMallocManaged(&edgeList, sizeof(uint) * (2* graph->num_edges + graph->num_nodes));


	uint *outDegreeCounter;
	uint source;
	uint end;
	
	long long counter=0;
	numParts = 0;
	int numZero = 0;
	
	for(int i=0; i<graph->num_nodes; i++)
	{
		nodePointer[i] = counter;
		edgeList[counter] = outDegree[i];
		
		if(outDegree[i] == 0)
			numZero++;
		
		if(outDegree[i] % Part_Size == 0)
			numParts += outDegree[i] / Part_Size ;
		else
			numParts += outDegree[i] / Part_Size + 1;
		
		counter = counter + outDegree[i] + 1;
	}

	cudaMallocManaged(&outDegreeCounter, sizeof(uint) * graph->num_nodes);
	
	for(int i=0; i<graph->num_edges; i++)
	{
		source = graph->edges[i].source;
		end = graph->edges[i].end;
		
		uint location = nodePointer[source]+1+outDegreeCounter[source];

		edgeList[location] = end;

		outDegreeCounter[source]++;  
	}
	
	cudaFree(outDegreeCounter);
	cudaMallocManaged(&partNodePointer, sizeof(PartPointer) * numParts);

	partNodePointer = new PartPointer[numParts];
	int thisNumParts;
	long long countParts = 0;
	for(int i=0; i<graph->num_nodes; i++)
	{
		if(outDegree[i] % Part_Size == 0)
			thisNumParts = outDegree[i] / Part_Size ;
		else
			thisNumParts = outDegree[i] / Part_Size + 1;
		for(int j=0; j<thisNumParts; j++)
		{
			partNodePointer[countParts].node = i;
			partNodePointer[countParts++].part = j;
		}
	}
}
