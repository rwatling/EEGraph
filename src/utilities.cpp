
#include "utilities.hpp"

void utilities::PrintResults(uint *results, uint n)
{
	cout << "Results of first "<< n << " nodes:\n[";
	for(int i=0; i<n; i++)
	{
		if(i>0)
			cout << " ";
		cout << i << ":" << results[i];
	}
	cout << "]\n";
}

void utilities::PrintResults(float *results, uint n)
{
	cout << "Results of first "<< n << " nodes:\n[";
	for(int i=0; i<n; i++)
	{
		if(i>0)
			cout << " ";
		cout << i << ":" << results[i];
	}
	cout << "]\n";
}

void utilities::SaveResults(string filepath, uint *results, uint n)
{
	cout << "Saving the results into the following file:\n";
	cout << ">> " << filepath << endl;
	ofstream outfile;
	outfile.open(filepath);
	for(int i=0; i<n; i++)
		outfile << i << " " << results[i] << endl;
	outfile.close();
	cout << "Done saving.\n";
}

void utilities::SaveResults(string filepath, float *results, uint n)
{
	cout << "Saving the results into " << filepath << " ...... " << flush;
	ofstream outfile;
	outfile.open(filepath);
	for(int i=0; i<n; i++)
		outfile << i << " " << results[i] << endl;
	outfile.close();
	cout << " Completed.\n";
}

void utilities::CompareArrays(unsigned int* arr1, unsigned int* arr2, int n) {
	for (int i = 0; i < n; i++) {
		if (arr1[i] != arr2[i]) {
			cout << "Arrays begin to differ at element " << i << endl;
			cout << "Arr1[" << i << "]: " << arr1[i] << endl;
			cout << "Arr2[" << i << "]: " << arr2[i] << endl;
			break;
		}
	}
}

double utilities::maxActivePct(vector<unsigned int> activeNodes, unsigned int total){
	unsigned int max = 0;
	for (unsigned int itrActive : activeNodes) {
		if (itrActive > max) {
			max = itrActive;
		}
	}
	return (double) max / (double) total;
}

double utilities::pctIterOverThreshold(vector<unsigned int> activeNodes, unsigned int total, double threshold){
	unsigned int count = 0;
	for (unsigned int itrActive : activeNodes) {
		double itrPct = (double) itrActive / (double) total;
		if (itrPct > threshold) {
			count++;
		}
	}
	return (double) count / (double) activeNodes.size();
}