#ifndef KERNELS_H

#define KERNELS_H

__global__ void gpuSUMReduction(const float *dA, float *dsumm, const int N);

__global__ void gpuSUMReductionWithoutDivergence(const float *dA, float *dsumm, const int N);

__global__ void gpuSUMReductionSeqAddress(const float *dA, float *dsumm, const int N);

__global__ void gpuSUMReductionFirstAdd(const float *dA, float *dsumm, const int N);

__global__ void gpuSUMReductionWarpReduce(const float *dA, float *dsumm, const int N);


template <const int blockSize>
__global__ void gpuSUMReductionCompleteUnroll(const float*, float*, const int);

template <const int blockSize>
__global__ void gpuSUMReductionMultiAddPerThread(const float*, float*, const int);

template <const int blockSize>
//__global__ void gpuSUMReductionCustom1(const float *dA, float *dsumm, const int N);
__global__ void gpuSUMReductionCustom1(const float *__restrict__ dA, float *__restrict__ dsumm, const int N);


#endif
