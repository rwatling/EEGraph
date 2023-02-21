#ifndef SUBWAY_ARGUMENT_PARSING_HPP
#define SUBWAY_ARGUMENT_PARSING_HPP

#include "subway_globals.hpp"


class SubwayArgumentParser
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
	string input;
	int sourceNode;
	string output;
	int deviceID;
	int numberOfItrs;
	
	bool energy;
	bool hasEnergyFile;
	bool hasEnergyStats;
	string energyFile;
	string energyStats;

	int benchmark;
	
	SubwayArgumentParser(int argc, char **argv, bool canHaveSource, bool canHaveItrs);
	
	bool Parse();
	
	string GenerateHelpString();
	
};


#endif	//	SUBWAY_ARGUMENT_PARSING_HPP
