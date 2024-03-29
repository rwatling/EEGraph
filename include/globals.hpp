#ifndef GLOBALS_HPP
#define GLOBALS_HPP

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

using namespace std;

typedef unsigned int uint;

const unsigned int Part_Size = 8;

const unsigned int DIST_INFINITY = UINT_MAX - 1;

const unsigned int RANDOM_SEED = 10293847;

enum Variant : unsigned char {
    ASYNC_PUSH_TD,
    ASYNC_PUSH_DD,
    SYNC_PUSH_TD,
    SYNC_PUSH_DD,
};

struct Edge{
    unsigned int source;
    unsigned int end;
};

struct PartPointer{
	unsigned int node;
	unsigned int part;
};

struct Result{
    float time;
    double energy;
    vector<unsigned int> sumLabelsVec;
};


#endif 	//	GLOBALS_HPP
