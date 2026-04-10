#ifndef KERNELS_H

#define KERNELS_H

__global__ void gpuSUMReduction(const float *dA, float *dsumm, const int N);

__global__ void gpuSUMReductionWithoutDivergence(const float *dA, float *dsumm, const int N);

__global__ void gpuSUMReductionSeqAddress(const float *dA, float *dsumm, const int N);

#endif
