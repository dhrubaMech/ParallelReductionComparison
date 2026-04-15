#include <iostream>
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


__global__ void gpuSUMReductionFirstAdd(const float *dA, float *dsumm, const int N){

    extern __shared__ float dAshared[];

    const int tid = threadIdx.x;
    const int globalIdx = (2*blockDim.x) * blockIdx.x + tid;

    // dAshared[tid] = globalIdx+blockDim.x < N ? dA[globalIdx] + dA[globalIdx+blockDim.x] : 0.0f;
    dAshared[tid]  = globalIdx < N ? dA[globalIdx] : 0.0f;
    dAshared[tid] += globalIdx+blockDim.x < N ? dA[globalIdx+blockDim.x] : 0.0f;
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

__device__ void warpReduce32(volatile float *dAshared, int tid){
    dAshared[tid] += dAshared[tid + 32];
    dAshared[tid] += dAshared[tid + 16];
    dAshared[tid] += dAshared[tid + 8];
    dAshared[tid] += dAshared[tid + 4];
    dAshared[tid] += dAshared[tid + 2];
    dAshared[tid] += dAshared[tid + 1];
}

__global__ void gpuSUMReductionWarpReduce(const float *dA, float *dsumm, const int N){

    extern __shared__ float dAshared[];

    const int tid = threadIdx.x;
    const int globalIdx = (2*blockDim.x) * blockIdx.x + tid;

    // dAshared[tid] = globalIdx+blockDim.x < N ? dA[globalIdx] + dA[globalIdx+blockDim.x] : 0.0f;
    dAshared[tid]  = globalIdx < N ? dA[globalIdx] : 0.0f;
    dAshared[tid] += globalIdx+blockDim.x < N ? dA[globalIdx+blockDim.x] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x/2 ; s > 32 ; s /= 2){
        if (tid < s){
            dAshared[tid] += dAshared[tid+s];
        }
        __syncthreads();
    }

    // manually doing loop unroll
    if (tid < 32){
       warpReduce32(dAshared,tid);
    }

    if (tid == 0){
        dsumm[blockIdx.x] = dAshared[0];
    }

}

template <const int blockSize>
__device__ void warpReduce(volatile float *dAshared, int tid){
    if (blockSize >= 64) dAshared[tid] += dAshared[tid + 32];
    if (blockSize >= 32) dAshared[tid] += dAshared[tid + 16];
    if (blockSize >= 16) dAshared[tid] += dAshared[tid + 8];
    if (blockSize >=  8) dAshared[tid] += dAshared[tid + 4];
    if (blockSize >=  4) dAshared[tid] += dAshared[tid + 2];
    if (blockSize >=  2) dAshared[tid] += dAshared[tid + 1];
}

template <const int blockSize>
__global__ void gpuSUMReductionCompleteUnroll(const float *dA, float *dsumm, const int N){

    extern __shared__ float dAshared[];

    const int tid = threadIdx.x;
    const int globalIdx = (2*blockDim.x) * blockIdx.x + tid;

    dAshared[tid]  = globalIdx < N ? dA[globalIdx] : 0.0f;
    dAshared[tid] += globalIdx+blockDim.x < N ? dA[globalIdx+blockDim.x] : 0.0f;
    // dAshared[tid] = globalIdx+blockDim.x < N ? dA[globalIdx] + dA[globalIdx+blockDim.x] : 0.0f;
    __syncthreads();
    
    // manually doing loop unroll
    if (blockSize >= 512){
        if (tid < 256){
            dAshared[tid] += dAshared[tid + 256];
	}
	__syncthreads();
    }
    if (blockSize >= 256){
        if (tid < 128){
            dAshared[tid] += dAshared[tid + 128];
        }
	__syncthreads();
    }
    if (blockSize >= 128){
        if (tid < 64){
            dAshared[tid] += dAshared[tid + 64];
        }
	__syncthreads();
    }
    

    if (tid < 32){
       warpReduce<blockSize>(dAshared,tid);
    }


    if (tid == 0){
        dsumm[blockIdx.x] = dAshared[0];
    }

}

template __global__ void gpuSUMReductionCompleteUnroll<16>(const float*, float*, const int);
template __global__ void gpuSUMReductionCompleteUnroll<32>(const float*, float*, const int);
template __global__ void gpuSUMReductionCompleteUnroll<64>(const float*, float*, const int);
template __global__ void gpuSUMReductionCompleteUnroll<128>(const float*, float*, const int);
template __global__ void gpuSUMReductionCompleteUnroll<256>(const float*, float*, const int);
template __global__ void gpuSUMReductionCompleteUnroll<512>(const float*, float*, const int);


template <const int blockSize>
__global__ void gpuSUMReductionMultiAddPerThread(const float *dA, float *dsumm, const int N){

    extern __shared__ float dAshared[];

    const int tid = threadIdx.x;
    int globalIdx = (2*blockDim.x) * blockIdx.x + tid;
    const int gridSize = 2*blockDim.x * gridDim.x;

    dAshared[tid] = 0.0f;
    while (globalIdx < N){
        //dAshared[tid] += dA[globalIdx] + dA[globalIdx + blockSize]; 
        // if (tid == 0 && blockIdx.x == 0) { printf("tid %d | globalIdx %d | grisize %d | gridDim.x %d | blockDim.x %d |\n",tid,globalIdx,gridSize,gridDim.x,blockDim.x); }
        dAshared[tid] += dA[globalIdx];
	dAshared[tid] += (globalIdx+blockSize) < N ? dA[globalIdx+blockSize] : 0.0f;
	globalIdx += gridSize;
    }
    __syncthreads();

    // manually doing loop unroll
    if (blockSize >= 512){
        if (tid < 256){
            dAshared[tid] += dAshared[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256){
        if (tid < 128){
            dAshared[tid] += dAshared[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128){
        if (tid < 64){
            dAshared[tid] += dAshared[tid + 64];
        }
        __syncthreads();
    }


    if (tid < 32){
       warpReduce<blockSize>(dAshared,tid);
    }


    if (tid == 0){
        dsumm[blockIdx.x] = dAshared[0];
    }

}

template __global__ void gpuSUMReductionMultiAddPerThread<16>(const float*, float*, const int);
template __global__ void gpuSUMReductionMultiAddPerThread<32>(const float*, float*, const int);
template __global__ void gpuSUMReductionMultiAddPerThread<64>(const float*, float*, const int);
template __global__ void gpuSUMReductionMultiAddPerThread<128>(const float*, float*, const int);
template __global__ void gpuSUMReductionMultiAddPerThread<256>(const float*, float*, const int);
template __global__ void gpuSUMReductionMultiAddPerThread<512>(const float*, float*, const int);



template <const int blockSize>
//__global__ void gpuSUMReductionCustom1(const float *dA, float *dsumm, const int N){
__global__ void gpuSUMReductionCustom1(const float *__restrict__ dA, float *__restrict__ dsumm, const int N){
    
    extern __shared__ float dAshared[];

    const int tid = threadIdx.x;
    int globalIdx = (2*blockDim.x) * blockIdx.x + tid;
    const int gridSize = 2*blockDim.x * gridDim.x;
    
    float summ = 0.0f;
    while (globalIdx < N){
        summ += dA[globalIdx];
	summ += (globalIdx + blockSize) < N ? dA[globalIdx + blockSize] : 0.0f;
        globalIdx += gridSize;
    }
    dAshared[tid] = summ;
    __syncthreads();


    // manually doing loop unroll
    if (blockSize >= 512){
        if (tid < 256){
            dAshared[tid] += dAshared[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256){
        if (tid < 128){
            dAshared[tid] += dAshared[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128){
        if (tid < 64){
            dAshared[tid] += dAshared[tid + 64];
        }
        __syncthreads();
    }
    if (blockSize >= 64){
        if (tid < 32){
            dAshared[tid] += dAshared[tid + 32];
        }
        __syncthreads();
    }

    summ = dAshared[tid];
    if (tid < 32){
       unsigned mask = __activemask();  // for safety only active threads are identified
       #pragma unroll
       for (int offset = 16; offset > 0; offset /= 2){
           // summ += __shfl_down_sync(0xffffffff, summ, offset);
           summ += __shfl_down_sync(mask, summ, offset);
       }
       /*
       #pragma unroll
       for (int offset = 1; offset < 32; offset *= 2){
           summ += __shfl_up_sync(0xffffffff, summ, offset);
       }
       */
       /*
       summ += __shfl_down_sync(mask, summ, 16);
       summ += __shfl_down_sync(mask, summ,  8);
       summ += __shfl_down_sync(mask, summ,  4);
       summ += __shfl_down_sync(mask, summ,  2);
       summ += __shfl_down_sync(mask, summ,  1);
       */
    }


    if (tid == 0){
        // dsumm[blockIdx.x] = dAshared[0];
        dsumm[blockIdx.x] = summ;
    }
}

template __global__ void gpuSUMReductionCustom1<16>(const float*, float*, const int);
template __global__ void gpuSUMReductionCustom1<32>(const float*, float*, const int);
template __global__ void gpuSUMReductionCustom1<64>(const float*, float*, const int);
template __global__ void gpuSUMReductionCustom1<128>(const float*, float*, const int);
template __global__ void gpuSUMReductionCustom1<256>(const float*, float*, const int);
template __global__ void gpuSUMReductionCustom1<512>(const float*, float*, const int);

