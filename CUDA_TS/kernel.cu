#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>

__global__ void MuSigma(float* expansionTS_d, float* mu_d, float* sigma_d, int subseqLen, int more) {
    
    for (int i = 0; i < more; i++) {
        int index = (threadIdx.x + blockIdx.x * blockDim.x) + (i * gridDim.x * blockDim.x);
        float sum = 0;
        for (int j = index; j < index + subseqLen; j++) {
            sum += expansionTS_d[j];
        }
        mu_d[index] = sum / subseqLen;
        sum = 0;
        for (int j = index; j < index + subseqLen; j++) {
            sum += pow(expansionTS_d[j] - mu_d[index], 2);
        }
        sigma_d[index] = sqrtf(sum / subseqLen);
    }  
}

__global__ void Scalar(float* TS_d, float* scalar_d, int* shift_d, int* x_d, int subseqLen, int more, int scalarLen, int matrixLen) {

    for (int i = 0; i < more; i++) {
        int index_x = (threadIdx.x + blockIdx.x * blockDim.x) + (i * gridDim.x * blockDim.x);
        if (index_x < scalarLen) {
            index_x = x_d[(threadIdx.x + blockIdx.x * blockDim.x) + (i * gridDim.x * blockDim.x)];
            int x = index_x % matrixLen;
            int y = index_x / matrixLen;
            float sum = 0;
            for (int i = 0; i < subseqLen; i++) {
                sum += TS_d[x + i] * TS_d[y + i];
            }
            scalar_d[x + (y * matrixLen) - shift_d[y]] = sum;
        }
    }
}

extern "C" __declspec(dllexport) void cudaRan_MuSigma(float * ts, int sizeTS, int subseqLen, float* mu, float* sigma) {

    int sizeSubseq = sizeTS - subseqLen + 1;
    int block = 10;
    int thread = 100;
    int allTread = block * thread;
    int more = sizeSubseq / allTread + 1;
    int num = more * allTread;

    float *expansionTS = new float[num + subseqLen - 1] {} ;
    //float *mu = new float[num] {} ;
    //float *sigma = new float[num] {} ;
    
    for (int i = 0; i < sizeTS; i++) {
        expansionTS[i] = ts[i];
    }
    float* expansionTS_d;
    float* mu_d;
    float* sigma_d;

    cudaMalloc((void**)&expansionTS_d, sizeof(float) * (num + subseqLen - 1));
    cudaMalloc((void**)&mu_d, sizeof(float) * num);
    cudaMalloc((void**)&sigma_d, sizeof(float) * num);

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMemcpy(expansionTS_d, expansionTS, sizeof(float) * (num + subseqLen - 1), cudaMemcpyKind::cudaMemcpyHostToDevice);

    MuSigma <<<block, thread>>> (expansionTS_d, mu_d, sigma_d, subseqLen, more);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time work: %f\n", time);
    
    cudaMemcpy(mu, mu_d, sizeof(float) * num, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaMemcpy(sigma, sigma_d, sizeof(float) * num, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    cudaFree(expansionTS_d);
    cudaFree(mu_d);
    cudaFree(sigma_d);
}


extern "C" __declspec(dllexport) void cudaRan_Scalar(float* ts, int sizeTS, int subseqLen, float * scalar) {

    int sizeSubseq = sizeTS - subseqLen + 1;

    int block = 100;
    int thread = 100;
    int allTread = thread * block;

    int* shift = new int[sizeSubseq - 1] {1};
    for (int i = 2; i < sizeSubseq; i++) {
        shift[i - 1] = shift[i - 2] + i;
    }

    int more = shift[sizeSubseq - 2] / allTread + 1;

    int* buf = new int[sizeSubseq - 1] {0};
    for (int i = 1; i < sizeSubseq - 1; i++) {
        buf[i] = (sizeSubseq + 1) * i;
    }

    int* x = new int[shift[sizeSubseq - 2]];

    int io = 0;
    for (int i = sizeSubseq - 1; i > 0; i--) {
        for (int j = 0; j < i; j++) {
            x[io] = j + 1 + buf[sizeSubseq - 1 - i];
            io++;
        }
    }

    //float* scalar = new float[shift[sizeSubseq - 2]];
    
    float* TS_d;
    float* scalar_d;
    int* shift_d;
    int* x_d;

    cudaMalloc((void**)&TS_d, sizeof(float) * sizeTS);
    cudaMalloc((void**)&scalar_d, sizeof(float) * shift[sizeSubseq - 2]);
    cudaMalloc((void**)&shift_d, sizeof(int) * (sizeSubseq - 1));
    cudaMalloc((void**)&x_d, sizeof(int) * shift[sizeSubseq - 2]);

    float time;

    cudaMemcpy(TS_d, ts, sizeof(float) * sizeTS, cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(shift_d, shift, sizeof(int) * (sizeSubseq - 1), cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(x_d, x, sizeof(int) * shift[sizeSubseq - 2], cudaMemcpyKind::cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    Scalar <<<block, thread >>> (TS_d, scalar_d, shift_d, x_d, subseqLen, more, shift[sizeSubseq - 2], sizeSubseq);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time work: %f\n", time);

    cudaMemcpy(scalar, scalar_d, sizeof(float) * shift[sizeSubseq - 2], cudaMemcpyKind::cudaMemcpyDeviceToHost);

    cudaFree(TS_d);
    cudaFree(scalar_d);
    cudaFree(shift_d);
    cudaFree(x_d);
}
