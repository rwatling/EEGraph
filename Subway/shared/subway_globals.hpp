#ifndef SUBWAY_GLOBALS_HPP
#define SUBWAY_GLOBALS_HPP

#include <iostream>
#include <stdlib.h>
#include <ctime>
#include <fstream>
#include <string>
#include <ctime>
#include <random>
#include <stdio.h>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <math.h>
#include <chrono>
#include <stdexcept>
#include <iostream>
#include <sstream> 

using namespace std;

#ifndef GLOBALS_HPP
const unsigned int DIST_INFINITY = std::numeric_limits<unsigned int>::max() - 1;
#endif

typedef unsigned int uint;
typedef unsigned long long ull;


struct OutEdge{
    uint end;
};

struct OutEdgeWeighted{
    uint end;
    uint w8;
};

#ifndef GLOBALS_HPP //To avoid conflicts with EEGraph Globals
struct Edge{
	uint source;
    uint end;
};
#endif

struct EdgeWeighted{
	uint source;
    uint end;
    uint w8;
};




#endif 	//	SUBWAY_GLOBALS_HPP
