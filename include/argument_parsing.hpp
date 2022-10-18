#ifndef ARGUMENT_PARSING_HPP
#define ARGUMENT_PARSING_HPP

#include "globals.hpp"

class ArgumentParser
{
private:

public:
	int argc;
	char** argv;
	
	bool canHaveSource;
	bool canHaveItrs;
	
	bool hasInput;
	bool hasSourceNode;
	bool hasOutput;
	bool hasDeviceID;
	bool hasNumberOfItrs;
	bool hasEnergyFile;
	bool debug;
	bool energy;

	string input;
	string output;
	string energyFile;

	int sourceNode;
	int deviceID;
	int numberOfItrs;

	Variant variant;
	
	ArgumentParser(int argc, char **argv, bool canHaveSource, bool canHaveItrs);
	
	bool Parse();
	
	string GenerateHelpString();
	
};


#endif	//	ARGUMENT_PARSING_HPP
