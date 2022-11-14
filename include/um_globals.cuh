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

typedef unsigned int uint;

const unsigned int Part_Size = 8;

const unsigned int DIST_INFINITY = UINT_MAX - 1;

enum Variant : unsigned char {
    ASYNC_PUSH_TD,
    ASYNC_PUSH_DD,
    ASYNC_PULL_TD,
    ASYNC_PULL_DD,
    SYNC_PUSH_TD,
    SYNC_PUSH_DD,
    SYNC_PULL_TD,
    SYNC_PULL_DD,
};

struct Edge : public Managed
{
    unsigned int source;
    unsigned int end;
};

struct PartPointer : public Managed
{
	unsigned int node;
	unsigned int part;
};


#endif 	//	UM_GLOBALS_CUH
