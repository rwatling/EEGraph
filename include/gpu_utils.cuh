#ifndef GPU_UTILS_H
#define GPU_UTILS_1
#include "cuda_includes.cuh"

__global__ void clearLabel(bool *label, unsigned int size);

__global__ void mixLabels(bool *label1, bool *label2, unsigned int size);

__global__ void moveUpLabels(bool *label1, bool *label2, unsigned int size);

#endif //GPU_UTILS_H