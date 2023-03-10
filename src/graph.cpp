
#include "../include/graph.hpp"

Graph::Graph(string graphFilePath, bool isWeighted)
{
	this->graphFilePath = graphFilePath;
	this->isWeighted = isWeighted;
	graphLoaded = false;
	hasZeroID = false;
}

void Graph::ReadGraph()
{

	cout << "Reading the input graph from the following file:\n>> " << graphFilePath << endl;

	this->graphFormat = getFileExtension(graphFilePath);

	if(graphFormat == "bcsr" || graphFormat == "bwcsr") {
		ifstream infile (graphFilePath, ios::in | ios::binary);
	
		infile.read ((char*)&num_nodes, sizeof(uint));
		infile.read ((char*)&num_edges, sizeof(uint));
		
		uint* nodePointer = new uint[num_nodes+1];
		Edge* tempEdgelist = (Edge*) malloc((num_edges) * sizeof(Edge));
		
		infile.read ((char*)nodePointer, sizeof(uint)*num_nodes);
		infile.read ((char*)tempEdgelist, sizeof(Edge)*num_edges);

		edges.insert(edges.end(), &tempEdgelist[0], &tempEdgelist[num_edges]);

		free(nodePointer);
		free(tempEdgelist);
	} else if (graphFormat == "edges" || graphFormat == "el" || 
				graphFormat == "wel" || graphFormat == "txt" || graphFormat == "edgelist") {	

		ifstream infile;
		infile.open(graphFilePath);
		
		stringstream ss;
		
		uint max = 0;

		if(graphLoaded == true)
		{
			edges.clear();
			weights.clear();
		}	
		
		graphLoaded = true;

		uint source;
		uint end;
		uint w8;
		uint i = 0;
		
		string line;
		
		Edge newEdge;
		
		unsigned long edgeCounter = 0;
		
		while(getline( infile, line ))
		{
			if(line[0] < '0' || line[0] > '9')
				continue;
				
			ss.str("");
			ss.clear();
			ss << line;
			
			ss >> newEdge.source;
			ss >> newEdge.end;
			
			edges.push_back(newEdge);
			
			if (newEdge.source == 0)
				hasZeroID = true;
			if (newEdge.end == 0)
				hasZeroID = true;			
			if(max < newEdge.source)
				max = newEdge.source;
			if(max < newEdge.end)
				max = newEdge.end;
			
			if (isWeighted)
			{
				if (ss >> w8)
					weights.push_back(w8);
				else
					weights.push_back(1);
			}
			
			edgeCounter++;
		}
		
		infile.close();
		
		graphLoaded = true;
		
		num_edges = edgeCounter;
		num_nodes = max;
		if (hasZeroID)
			num_nodes++;
	} else {
		cout << "Graph file type not recognized" << endl;
	}

	cout << "Done reading.\n";
	cout << "Number of nodes = " << num_nodes << endl;
	cout << "Number of edges = " << num_edges << endl;
}

string Graph::getFileExtension(string fileName)
{
    if(fileName.find_last_of(".") != string::npos)
        return fileName.substr(fileName.find_last_of(".")+1);
    return "";
}

VertexSubgraph::VertexSubgraph(Graph &graph, string graphFilePath, bool isWeighted) : Graph(graphFilePath, isWeighted) {
	this->parentGraph = &graph;
}

void VertexSubgraph::MakeSubgraph(float pct, int sourceNode) {
	this->graphLoaded = parentGraph->graphLoaded;
	this->pct = pct;

	if (this->graphLoaded == false) {
		cout << "Graph has not been loaded" << endl;
		return;
	}

	this->subgraph_num_nodes = this->subgraph_num_nodes = pct * parentGraph->num_nodes;
	this->num_nodes = parentGraph->num_nodes;

	//Set up selected array
	selected = new bool[this->num_nodes];
	for (unsigned int i = 0; i < this->num_nodes; i++) {
		selected[i] = false;
	}
	selected[sourceNode] = true;

	//Select nodes
	// First 1/2 of subgraph nodes contain first 1/5 of full graph nodes
	// Therefore 0.5 * subgraph nodes < ~20% of full graph, so pct arg < 40%
	unsigned int count = 1;
	srand(RANDOM_SEED);
	while (count < this->subgraph_num_nodes) {
		unsigned int rand_node;
		if (count < (this->subgraph_num_nodes / 2)) {
			rand_node = (rand() % (parentGraph->num_nodes / 5));
		} else {
			rand_node = (rand() % (parentGraph->num_nodes));
		}
		
		if (!selected[rand_node]) {
			selected[rand_node] = true;
			count++;
		}
	}

	//Select induced edges
	this->num_edges = 0;
	uint max;
	for (unsigned int i = 0; i < parentGraph->num_edges; i++) {
		Edge newEdge;
		unsigned int source = parentGraph->edges[i].source;
		unsigned int end = parentGraph->edges[i].end;
		unsigned int w8 = parentGraph->weights[i];

		if (selected[source] && selected[end]) {
			this->num_edges = this->num_edges + 1;

			newEdge.source = source;
			newEdge.end = end;

			if (source == 0)
				this->hasZeroID = true;
			if (end == 0)
				this->hasZeroID = true;
			
			if (isWeighted)
			{
				this->weights.push_back(w8);
			}

			this->edges.push_back(newEdge);
		}
	}

	cout << "Done generating subgraph.\n";
	cout << "Subgraph number of nodes = " << this->subgraph_num_nodes << endl;
	cout << "Virtual graph assumed number of nodes = " << this->num_nodes << endl;
	cout << "Subgraph number of edges = " << this->num_edges << endl;
}