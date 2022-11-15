#ifndef UM_GLOBALS_CUH
#define UM_GLOBALS_CUH

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
#include <climits>

#include "managed.cuh"

using namespace std;

struct UMEdge : public Managed
{
    unsigned int source;
    unsigned int end;
};

struct UMPartPointer : public Managed
{
	unsigned int node;
	unsigned int part;
};


#endif 	//	UM_GLOBALS_CUH
