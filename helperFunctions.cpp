#include <stdio.h>
#include <random>
#include <iostream>

#include "helperFunctions.h"

using namespace std;

void fillRandomArray(float* a, const int N){
    static random_device rd;
    static mt19937 prng(rd());
    static uniform_real_distribution<float> dist(0.0f,1.0f);

    for (int i = 0 ; i < N ; i++){
        a[i] = dist(prng);
    }
}

void show1DArray(const float* a, const int N){
    for (int i = 0 ; i < N ; i++){
        printf("%f ",a[i]);
    }
    printf("\n");
}

float cpuSUMReduction(const float* A, const int N){
    float summ = 0.0f;
    for (int i = 0 ; i < N ; i++){
        summ += A[i];
    }
    return summ;
}


