#ifndef MANAGED_CUH
#define MANAGED_CUH 1
#include "gpu_error_check.cuh"
#include "cuda_includes.cuh"

class Managed {
public:
    void *operator new(size_t len) {
        void *ptr;
        gpuErrorcheck(cudaMallocManaged(&ptr, len));
        gpuErrorcheck(cudaDeviceSynchronize());
        return ptr;
    }

    void operator delete(void *ptr) {
        gpuErrorcheck(cudaDeviceSynchronize());
        gpuErrorcheck(cudaFree(ptr));
    }
};
#endif //MANAGED_CUH