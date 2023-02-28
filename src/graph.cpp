
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

/*void EdgeSubgraph::MakeSubgraph(Graph &graph, float pct) {
	this->parentGraph = &graph;
	graphLoaded = graph.graphLoaded;

	if (this->graphLoaded = false) {
		cerr << "Graph has not been loaded" << endl;
	}

	this->num_edges = pct * num_edges;
	srand(RAND_SEED);

	Edge newEdge;
	uint max=0;
	selected = new bool[this->num_edges];

	for (unsigned int i = 0; i < this->num_edges; i++) {
		selected[i] = false;
	}

	for (unsigned int i = 0; i < this->num_edges; i++) {
		unsigned int rand_edge;
		bool add = false;
		while (!add) {
			rand_edge = (rand() % graph.num_edges);

			if (selected[rand_edge]) {
				add = false;
			} else {
				add = true;
			}
		}

		uint source = graph.edges[rand_edge].source;
		uint end = graph.edges[rand_edge].end;
		uint w8 = graph.weights[rand_edge];

		newEdge.source = source;
		newEdge.end = end;

		this->edges.push_back(newEdge);

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
			weights.push_back(w8);
		}
	}

	this->num_nodes = max;
	cout << "Done generating subgraph.\n";
	cout << "Number of nodes = " << this->num_nodes << endl;
	cout << "Number of edges = " << this->num_edges << endl;
}*/