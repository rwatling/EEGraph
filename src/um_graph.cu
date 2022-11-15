
#include "../include/um_graph.cuh"

UMGraph::UMGraph(string graphFilePath, bool isWeighted)
{
	this->graphFilePath = graphFilePath;
	this->isWeighted = isWeighted;
	graphLoaded = false;
	hasZeroID = false;
}

void UMGraph::ReadGraph()
{

	cout << "Reading the input graph from the following file:\n>> " << graphFilePath << endl;

	this->graphFormat = getFileExtension(graphFilePath);

	if (graphFormat == "edges" || graphFormat == "el" || graphFormat == "wel") {	

		ifstream infile;
		infile.open(graphFilePath);
		
		stringstream ss;
		
		uint max = 0;
		vector<Edge> temp_edges;
		vector<uint> temp_weights;

		if(graphLoaded == true)
		{
			temp_edges.clear();
			temp_weights.clear();
		}	
		
		graphLoaded = true;

		uint w8;
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
			
			temp_edges.push_back(newEdge);
			
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
					temp_weights.push_back(w8);
				else
					temp_weights.push_back(1);
			}
			
			edgeCounter++;
		}
		
		infile.close();
		
		graphLoaded = true;
		
		num_edges = edgeCounter;
		num_nodes = max;
		if (hasZeroID)
			num_nodes++;

		cudaMallocManaged(&edges, num_edges * sizeof(Edge));
		cudaMallocManaged(&weights, num_edges * sizeof(unsigned int));

		copy(temp_edges.begin(), temp_edges.end(), edges);
		copy(temp_weights.begin(), temp_weights.end(), weights);
	} else {
		cout << "Graph file type not recognized" << endl;
	}

	cout << "Done reading.\n";
	cout << "Number of nodes = " << num_nodes << endl;
	cout << "Number of edges = " << num_edges << endl;
}

string UMGraph::getFileExtension(string fileName)
{
    if(fileName.find_last_of(".") != string::npos)
        return fileName.substr(fileName.find_last_of(".")+1);
    return "";
}
