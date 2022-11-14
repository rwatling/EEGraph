#include "../include/gpu_utils.cuh"

__global__ void clearLabel(bool *label, unsigned int size)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < size)
		label[id] = false;
}

__global__ void mixLabels(bool *label1, bool *label2, unsigned int size)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < size){
		label1[id] = label1[id] || label2[id];
		label2[id] = false;	
	}
}

__global__ void moveUpLabels(bool *label1, bool *label2, unsigned int size)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < size){
		label1[id] = label2[id];
		label2[id] = false;	
	}
}