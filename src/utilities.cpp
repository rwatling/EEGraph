
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
			break;
		}
	}
}