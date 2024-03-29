#include "../include/argument_parsing.hpp"
    
ArgumentParser::ArgumentParser(int argc, char **argv, bool canHaveSource, bool canHaveItrs)
{
	this->argc = argc;
	this->argv = argv;
	this->canHaveSource = canHaveSource;
	this->canHaveItrs = canHaveItrs;
	
	this->sourceNode = 0;
	this->deviceID = 0;
	this->numberOfItrs = 1;
	this->acc = 0.01;
	
	hasInput = false;
	hasSourceNode = false;
	hasOutput = false;
	hasDeviceID = false;
	hasNumberOfItrs = false;
	hasEnergyFile = false;
	debug = false;
	variant = ASYNC_PUSH_TD;
	energy = false;
	unifiedMem = false;
	hasAcc = false;
	nodeActivity = false;

	Parse();
}
	
bool ArgumentParser::Parse()
{
	try
	{
		if(argc == 1)
		{
			cout << GenerateHelpString();
			exit(0);
		}
		
		if(argc == 2) 
			if ((strcmp(argv[1], "--help") == 0) || 
				(strcmp(argv[1], "-help") == 0) || 
				(strcmp(argv[1], "--h") == 0) || 
				(strcmp(argv[1], "-h") == 0))
			{
				cout << GenerateHelpString();
				exit(0);
			}
		
		if(argc%2 == 0)
		{
			cout << "\nThere was an error parsing command line arguments\n";
			cout << GenerateHelpString();
			exit(0);
		}
		
			
		for(int i=1; i<argc-1; i=i+2)
		{
			
			if (strcmp(argv[i], "--input") == 0) {
				input = string(argv[i+1]);
				hasInput = true;
			}
			else if (strcmp(argv[i], "--output") == 0) {
				output = string(argv[i+1]);
				hasOutput = true;
			} else if (strcmp(argv[i], "--efile") == 0) {
				energyFile = string(argv[i+1]);
				hasEnergyFile = true;
			} else if (strcmp(argv[i], "--estats") == 0) {
				energyStats = string(argv[i+1]);
				hasEnergyStats = true;
			} else if (strcmp(argv[i], "--source") == 0 && canHaveSource) {
				sourceNode = atoi(argv[i+1]);
				hasSourceNode = true;
			} else if (strcmp(argv[i], "--device") == 0) {
				deviceID = atoi(argv[i+1]);
				hasDeviceID = true;
			} else if (strcmp(argv[i], "--iteration") == 0 && canHaveItrs) {
				numberOfItrs = atoi(argv[i+1]);
				hasNumberOfItrs = true;
			} else if (strcmp(argv[i], "--debug") == 0) {
				if (strcmp(argv[i+1], "true") == 0 || 
					strcmp(argv[i+1], "True") == 0 || 
					strcmp(argv[i+1], "TRUE") == 0)
					debug = true;
			} else if (strcmp(argv[i], "--activity") == 0) {
				if (strcmp(argv[i+1], "true") == 0 || 
					strcmp(argv[i+1], "True") == 0 || 
					strcmp(argv[i+1], "TRUE") == 0)
					nodeActivity = true;
			} else if (strcmp(argv[i], "--variant") == 0) {
				if (strcmp(argv[i+1], "async_push_td") == 0) {
					variant = ASYNC_PUSH_TD;
				} else if (strcmp(argv[i+1], "async_push_dd") == 0) {
					variant = ASYNC_PUSH_DD;
				} else if (strcmp(argv[i+1], "sync_push_td") == 0) {
					variant = SYNC_PUSH_TD;
				} else if (strcmp(argv[i+1], "sync_push_dd") == 0) {
					variant = SYNC_PUSH_DD;
				} else {
					cout << "Variant not recognized\n";
					exit(0);
				}
			} else if (strcmp(argv[i], "--energy") == 0) {
				if (strcmp(argv[i+1], "true") == 0 || 
					strcmp(argv[i+1], "True") == 0 || 
					strcmp(argv[i+1], "TRUE") == 0)
					energy = true;
			} else if (strcmp(argv[i], "--um") == 0) {
				if (strcmp(argv[i+1], "true") == 0 || 
					strcmp(argv[i+1], "True") == 0 || 
					strcmp(argv[i+1], "TRUE") == 0)
					unifiedMem = true; 
			} else if (strcmp(argv[i], "--accuracy") == 0) {
				acc = (float) atof(argv[i+1]);
				hasAcc = true;
			} else {
				cout << "\nThere was an error parsing command line argument <" << argv[i] << ">\n";
				cout << GenerateHelpString();
				exit(0);
			}
		}

		if(energy && (!hasEnergyFile || !hasEnergyStats)) {
			cout << "The option --energy was true but energy file and/or energy stats files were not included\n";
			exit(0);
		}
		
		if(hasInput)
			return true;
		else
		{
			cout << "\nInput graph file argument is required.\n";
			cout << GenerateHelpString();
			exit(0);
		}
	}
	catch( const std::exception& strException ) {
		std::cerr << strException.what() << "\n";
		GenerateHelpString();
		exit(0);
	}
	catch(...) {
		std::cerr << "An exception has occurred.\n";
		GenerateHelpString();
		exit(0);
	}
}

string ArgumentParser::GenerateHelpString(){
	string str = "\nRequired arguments:";
	str += "\n    [--input]: Input graph file. E.g., --input FacebookGraph.txt";
	str += "\nOptional arguments";
	if(canHaveSource)
		str += "\n    [--source]:  Begins from the source (Default: 0). E.g., --source 10";
	str += "\n    [--output]: Output file for results. E.g., --output results.txt";
	str += "\n    [--device]: Select GPU device (Default: 0). E.g., --device 1";
	if(canHaveItrs)
		str += "\n    [--iteration]: Number of iterations (Default: 1). E.g., --iterations 10";
	str += "\n    [--debug]: Check or observe information (Default: false). E.g. --debug true";
	str += "\n    [--variant]: Algorithm variant option(Default: async_push_td). E.g. --variant async_push_td";
	str += "\n    [--algorithm]: Algoriothm to run (Default: sssp). E.g. --algorithm bfs";
	str += "\n    [--energy]: Measure and output GPU energy information (Default: false). E.g. --energy true";
	str += "\n    [--efile]: Output file for energy (Required if energy == true). E.g. --efile my_experiment_energy";
	str += "\n    [--estats]: Output file for energy (Required if energy == true). E.g. --estats my_experiment_stats";
	str += "\n    [--um]: Use unified memory for graph algorithms. E.g. --um true";
	str += "\n    [--large]: Supplied graph is large so do not run classic. E.g. --large true";
	str += "\n    [--acc]: supply accuracy for algorithm (PageRank) E.g. --accuracy 0.01";
	str += "\n    [--activity]: sum node activity vector in results(Default false) E.g. --activity true";
	str += "\n\n";
	return str;
}

