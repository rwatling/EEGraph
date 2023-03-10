#ifndef UTILITIES_HPP
#define UTILITIES_HPP


#include "globals.hpp"

namespace utilities {
	void PrintResults(uint *results, uint n);
	void PrintResults(float *results, uint n);
	void SaveResults(string filepath, uint *results, uint n);
	void SaveResults(string filepath, float *results, uint n);
	void CompareArrays(unsigned int* arr1, unsigned int* arr2, int n);
	void CompareArrays(float* arr1, float* arr2, int n);
	double maxActivePct(vector<unsigned int> activeNodes, unsigned int total);
	double pctIterOverThreshold(vector<unsigned int> activeNodes, unsigned int total, double threshold);
}

#endif	//	UTILITIES_HPP
