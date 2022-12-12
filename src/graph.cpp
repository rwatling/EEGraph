
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

	srand(RAND_SEED);

	if(graphFormat == "bcsr" || graphFormat == "bwcsr") {
		ifstream infile (graphFilePath, ios::in | ios::binary);
	
		infile.read ((char*)&num_nodes, sizeof(uint));
		infile.read ((char*)&num_edges, sizeof(uint));
		
		uint* nodePointer = new uint[num_nodes+1];	//This is extra but should hopefully maintain correctness from Subway
		Edge* tempEdgelist = (Edge*) malloc((num_edges) * sizeof(Edge));
		
		infile.read ((char*)nodePointer, sizeof(uint)*num_nodes);
		infile.read ((char*)tempEdgelist, sizeof(Edge)*num_edges);

		edges.insert(edges.end(), &tempEdgelist[0], &tempEdgelist[num_edges]);

		free(nodePointer);
		free(tempEdgelist);
	} else if (graphFormat == "edges" || graphFormat == "el" || graphFormat == "wel") {	

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
					weights.push_back((rand() % RAND_RANGE) + 1);
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
