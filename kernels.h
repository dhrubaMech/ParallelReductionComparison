#ifndef KERNELS_H

#define KERNELS_H

__global__ void gpuSUMReduction(const float *dA, float *dsumm, const int N);

__global__ void gpuSUMReductionWithoutDivergence(const float *dA, float *dsumm, const int N);

__global__ void gpuSUMReductionSeqAddress(const float *dA, float *dsumm, const int N);

__global__ void gpuSUMReductionFirstAdd(const float *dA, float *dsumm, const int N);

__global__ void gpuSUMReductionWarpReduce(const float *dA, float *dsumm, const int N);


template <const int blockSize>
__global__ void gpuSUMReductionCompleteUnroll(const float*, float*, const int);


#endif
