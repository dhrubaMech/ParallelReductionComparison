#include <cuda.h>

#include "kernels.h"

__global__ void gpuSUMReduction(const float *dA, float *dsumm, const int N){

    extern __shared__ float dAshared[];

    const int tid = threadIdx.x;
    const int globalIdx = blockDim.x * blockIdx.x + tid;

    dAshared[tid] = globalIdx < N ? dA[globalIdx] : 0.0f;
    __syncthreads();

    for (int s = 1 ; s < blockDim.x ; s *= 2){
	if (tid%(s*2) == 0){
        // if ((tid%(s*2) == 0) && (tid+s < blockDim.x)){
	    dAshared[tid] += dAshared[tid+s];
	}
	__syncthreads();
    }
    if (tid == 0){
        dsumm[blockIdx.x] = dAshared[0];
    }
    
}


__global__ void gpuSUMReductionWithoutDivergence(const float *dA, float *dsumm, const int N){

    extern __shared__ float dAshared[];

    const int tid = threadIdx.x;
    const int globalIdx = blockDim.x * blockIdx.x + tid;

    dAshared[tid] = globalIdx < N ? dA[globalIdx] : 0.0f;
    __syncthreads();

    for (int s = 1 ; s < blockDim.x ; s *= 2){
        int index = 2 * s * tid;
        if (index + s < blockDim.x){
            dAshared[index] += dAshared[index+s];
        }
        __syncthreads();
    }
    if (tid == 0){
        dsumm[blockIdx.x] = dAshared[0];
    }

}


__global__ void gpuSUMReductionSeqAddress(const float *dA, float *dsumm, const int N){

    extern __shared__ float dAshared[];

    const int tid = threadIdx.x;
    const int globalIdx = blockDim.x * blockIdx.x + tid;

    dAshared[tid] = globalIdx < N ? dA[globalIdx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x/2 ; s > 0 ; s /= 2){
        if (tid < s){
            dAshared[tid] += dAshared[tid+s];
        }
        __syncthreads();
    }
    if (tid == 0){
        dsumm[blockIdx.x] = dAshared[0];
    }

}

