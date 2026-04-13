#include <stdio.h>
#include <chrono>
#include <iostream>
#include <fstream>

#include <cuda.h>

#include "helperFunctions.h"
#include "kernels.h"

using namespace std;

#define CHECK_CUDA(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
}

#define SM 256
#define WPT 4

int main(){
    
    const int repeat = 1;
    const int N = 1000000;

    const bool reduce0 = false ;
    const bool reduce1 = false ;
    const bool reduce2 = false ;
    const bool reduce3 = false ;
    const bool reduce4 = false ;
    const bool reduce5 = false ;
    const bool reduce6 = true ;

    float *A = new float[N];
    fillRandomArray(A,N);

    auto start = std::chrono::steady_clock::now();
    float summ = cpuSUMReduction(A,N);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::micro> Tspan = end - start;
    printf("Time taken [CPU] ~ %f microsecs\n",Tspan.count());
    printf("CPU sum : %f\n\n",summ);

    float *dA;
    cudaMalloc((void**) &dA, N * sizeof(float));
    cudaMemcpy(dA, A, N * sizeof(float), cudaMemcpyHostToDevice);

    if (reduce0){
        ofstream file("KernelTimings/reduce0_N"+to_string(N)+"_repeat"+to_string(repeat)+"_NT"+to_string(SM)+".csv");
        for (int r = 0 ; r < repeat ; r++){
            const int NT = SM;
            const int NB = (N + NT - 1)/NT;
	    // printf("NT %d | NB %d\n",NT,NB);

            float *dsumm;
	    cudaMalloc((void**) &dsumm, NB * sizeof(float));
	    cudaMemset(dsumm, 0.0f, NB * sizeof(float));
        
	    auto start = std::chrono::steady_clock::now();
	    gpuSUMReduction<<<NB,NT,NT*sizeof(float)>>>(dA,dsumm,N);
            cudaDeviceSynchronize();
	    auto  end = std::chrono::steady_clock::now();

	    std::chrono::duration<double, std::micro> Tspan = end - start;
	    printf("Time taken [GPU reduction] ~ %f microsecs\n",Tspan.count());
            file << Tspan.count() << ((r != repeat-1) ? "," : "");

            float hsumm[NB] = {0.0f};
	    cudaMemcpy(hsumm, dsumm, NB * sizeof(float), cudaMemcpyDeviceToHost); cudaFree(dsumm);

            float res = 0.0f;
	    for (int i = 0 ; i < NB ; i++){
		res += hsumm[i];
	    }
	    printf("Error [red0] : %f | [%f]\n\n",fabs(res-summ),res);
        }
        file.close();
    }


    if (reduce1){
        ofstream file("KernelTimings/reduce1_N"+to_string(N)+"_repeat"+to_string(repeat)+"_NT"+to_string(SM)+".csv");
        for (int r = 0 ; r < repeat ; r++){
            const int NT = SM;
            const int NB = (N + NT - 1)/NT;
            // printf("NT %d | NB %d\n",NT,NB);

            float *dsumm;
            cudaMalloc((void**) &dsumm, NB * sizeof(float));
            cudaMemset(dsumm, 0.0f, NB * sizeof(float));

            auto start = std::chrono::steady_clock::now();
            gpuSUMReductionWithoutDivergence<<<NB,NT,NT*sizeof(float)>>>(dA,dsumm,N);
            cudaDeviceSynchronize();
            auto  end = std::chrono::steady_clock::now();

            std::chrono::duration<double, std::micro> Tspan = end - start;
            printf("Time taken [GPU wihtout divergence] ~ %f microsecs\n",Tspan.count());
            file << Tspan.count() << ((r != repeat-1) ? "," : "");

            float hsumm[NB] = {0.0f};
            cudaMemcpy(hsumm, dsumm, NB * sizeof(float), cudaMemcpyDeviceToHost); cudaFree(dsumm);

            float res = 0.0f;
            for (int i = 0 ; i < NB ; i++){
                res += hsumm[i];
            }
            printf("Error [red1] : %f | [%f]\n\n",fabs(res-summ),res);
        }
        file.close();
    }

    if (reduce2){
        ofstream file("KernelTimings/reduce2_N"+to_string(N)+"_repeat"+to_string(repeat)+"_NT"+to_string(SM)+".csv");
        for (int r = 0 ; r < repeat ; r++){
            const int NT = SM;
            const int NB = (N + NT - 1)/NT;
            // printf("NT %d | NB %d\n",NT,NB);

            float *dsumm;
            cudaMalloc((void**) &dsumm, NB * sizeof(float));
            cudaMemset(dsumm, 0.0f, NB * sizeof(float));

            auto start = std::chrono::steady_clock::now();
            gpuSUMReductionSeqAddress<<<NB,NT,NT*sizeof(float)>>>(dA,dsumm,N);
            cudaDeviceSynchronize();
            auto  end = std::chrono::steady_clock::now();

            std::chrono::duration<double, std::micro> Tspan = end - start;
            printf("Time taken [Sqeuential Address] ~ %f microsecs\n",Tspan.count());
            file << Tspan.count() << ((r != repeat-1) ? "," : "");

            float hsumm[NB] = {0.0f};
            cudaMemcpy(hsumm, dsumm, NB * sizeof(float), cudaMemcpyDeviceToHost); cudaFree(dsumm);

            float res = 0.0f;
            for (int i = 0 ; i < NB ; i++){
                res += hsumm[i];
            }
            printf("Error [red2] : %f | [%f]\n\n",fabs(res-summ),res);
        }
        file.close();
    }

    if (reduce3){
        ofstream file("KernelTimings/reduce3_N"+to_string(N)+"_repeat"+to_string(repeat)+"_NT"+to_string(SM)+".csv");
        for (int r = 0 ; r < repeat ; r++){
            const int NT = SM;
            const int NB = (N + 2*NT - 1)/(2*NT);  // reducing the number of blocks launched
            // printf("NT %d | NB %d\n",NT,NB);

            float *dsumm;
            cudaMalloc((void**) &dsumm, NB * sizeof(float));
            cudaMemset(dsumm, 0.0f, NB * sizeof(float));

            auto start = std::chrono::steady_clock::now();
            gpuSUMReductionFirstAdd<<<NB,NT,NT*sizeof(float)>>>(dA,dsumm,N);
            cudaDeviceSynchronize();
            auto  end = std::chrono::steady_clock::now();

            std::chrono::duration<double, std::micro> Tspan = end - start;
            printf("Time taken [First Add] ~ %f microsecs\n",Tspan.count());
            file << Tspan.count() << ((r != repeat-1) ? "," : "");

            float hsumm[NB] = {0.0f};
            cudaMemcpy(hsumm, dsumm, NB * sizeof(float), cudaMemcpyDeviceToHost); cudaFree(dsumm);

            float res = 0.0f;
            for (int i = 0 ; i < NB ; i++){
                res += hsumm[i];
            }
            printf("Error [red3] : %f | [%f]\n\n",fabs(res-summ),res);
        }
        file.close();
    }


    if (reduce4){
        ofstream file("KernelTimings/reduce4_N"+to_string(N)+"_repeat"+to_string(repeat)+"_NT"+to_string(SM)+".csv");
        for (int r = 0 ; r < repeat ; r++){
            const int NT = SM;
            const int NB = (N + (2*NT) - 1)/(2*NT);  // reducing the number of blocks launched
            // printf("NT %d | NB %d\n",NT,NB);

            float *dsumm;
            cudaMalloc((void**) &dsumm, NB * sizeof(float));
            cudaMemset(dsumm, 0.0f, NB * sizeof(float));

            auto start = std::chrono::steady_clock::now();
            gpuSUMReductionWarpReduce<<<NB,NT,NT*sizeof(float)>>>(dA,dsumm,N);
            cudaDeviceSynchronize();
            auto  end = std::chrono::steady_clock::now();

            std::chrono::duration<double, std::micro> Tspan = end - start;
            printf("Time taken [Warp reduce] ~ %f microsecs\n",Tspan.count());
            file << Tspan.count() << ((r != repeat-1) ? "," : "");

            float hsumm[NB] = {0.0f};
            cudaMemcpy(hsumm, dsumm, NB * sizeof(float), cudaMemcpyDeviceToHost); cudaFree(dsumm);

            float res = 0.0f;
            for (int i = 0 ; i < NB ; i++){
                res += hsumm[i];
            }
            printf("Error [red4] : %f | [%f]\n\n",fabs(res-summ),res);
        }
        file.close();
    }


    if (reduce5){
        ofstream file("KernelTimings/reduce5_N"+to_string(N)+"_repeat"+to_string(repeat)+"_NT"+to_string(SM)+".csv");
        for (int r = 0 ; r < repeat ; r++){
            const int NT = SM;
            const int NB = (N + (2*NT) - 1)/(2*NT);  // reducing the number of blocks launched
            // printf("NT %d | NB %d\n",NT,NB);

            float *dsumm;
            cudaMalloc((void**) &dsumm, NB * sizeof(float));
            cudaMemset(dsumm, 0.0f, NB * sizeof(float));

	    auto start = std::chrono::steady_clock::now();
            gpuSUMReductionCompleteUnroll<NT><<<NB,NT,NT*sizeof(float)>>>(dA,dsumm,N);
            cudaDeviceSynchronize();
            auto  end = std::chrono::steady_clock::now();

            std::chrono::duration<double, std::micro> Tspan = end - start;
            printf("Time taken [Complete unroll] ~ %f microsecs\n",Tspan.count());
            file << Tspan.count() << ((r != repeat-1) ? "," : "");

            float hsumm[NB] = {0.0f};
            cudaMemcpy(hsumm, dsumm, NB * sizeof(float), cudaMemcpyDeviceToHost); cudaFree(dsumm);

            float res = 0.0f;
            for (int i = 0 ; i < NB ; i++){
                res += hsumm[i];
            }
            printf("Error [red5] : %f | [%f]\n\n",fabs(res-summ),res);
        }
        file.close();
    }

    
    if (reduce6){
        // ofstream file("KernelTimings/reduce6_N"+to_string(N)+"_repeat"+to_string(repeat)+"_NT"+to_string(SM)+".csv");
        ofstream file("KernelTimings/reduce6_N"+to_string(N)+"_repeat"+to_string(repeat)+"_NT"+to_string(SM)+"_WPT"+to_string(WPT)+".csv");
        for (int r = 0 ; r < repeat ; r++){
            const int NT = SM;
            const int NB = ((N + (2*NT) - 1)/(2*NT))/WPT;  // reducing the number of blocks launched
            // printf("NT %d | NB %d\n",NT,NB);

            float *dsumm;
            cudaMalloc((void**) &dsumm, NB * sizeof(float));
            cudaMemset(dsumm, 0.0f, NB * sizeof(float));

            auto start = std::chrono::steady_clock::now();
            gpuSUMReductionMultiAddPerThread<NT><<<NB,NT,NT*sizeof(float)>>>(dA,dsumm,N);
            cudaDeviceSynchronize();
            auto  end = std::chrono::steady_clock::now();

            std::chrono::duration<double, std::micro> Tspan = end - start;
            printf("Time taken [Multi Add/Thread] ~ %f microsecs\n",Tspan.count());
            file << Tspan.count() << ((r != repeat-1) ? "," : "");

            float hsumm[NB] = {0.0f};
            cudaMemcpy(hsumm, dsumm, NB * sizeof(float), cudaMemcpyDeviceToHost); cudaFree(dsumm);

            float res = 0.0f;
            for (int i = 0 ; i < NB ; i++){
                res += hsumm[i];
            }
            printf("Error [red6] : %f | [%f]\n\n",fabs(res-summ),res);
        }
        file.close();
    }






    delete[] A;
    cudaFree(dA);

    return 0;
}


