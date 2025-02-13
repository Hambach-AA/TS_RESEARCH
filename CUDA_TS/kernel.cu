#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <iostream>
#include <cmath>
#include <time.h> 


__global__ void Pre_GPU_MuSigma(double* ts_d, double* mu_d, double* sigma_d, int subseqLen, int batchIterations, int sizeSubseq) {
    
    for (int i = 0; i < batchIterations; i++) {
        int index = (threadIdx.x + blockIdx.x * blockDim.x) + (i * gridDim.x * blockDim.x);
        if (index < sizeSubseq) {
            double sum = 0;
            for (int j = index; j < index + subseqLen; j++) {
                sum += ts_d[j];
            }
            mu_d[index] = sum / subseqLen;
            sum = 0;
            for (int j = index; j < index + subseqLen; j++) {
                sum += (ts_d[j] - mu_d[index]) * (ts_d[j] - mu_d[index]);
            }
            sigma_d[index] = sqrt(sum / subseqLen);
        }
    }  
}

__global__ void Pre_GPU_ScalarProdPart1(double* ts_d, double* vector_d, int subseqLen, int batchIterations, int subseqNum) {

    for (int i = 0; i < batchIterations; i++) {
        int index = (threadIdx.x + blockIdx.x * blockDim.x) + (i * gridDim.x * blockDim.x);
        if (index < subseqNum) {
            int x = index % subseqNum;
            int y = index / subseqNum;
            if (x != y) {
                double sum = 0;
                for (int i = 0; i < subseqLen; i++) {
                    sum += ts_d[x + i] * ts_d[y + i];
                }
                vector_d[x + (y * subseqNum)] = sum;
                vector_d[y + (x * subseqNum)] = sum;
            }
        }
    }
}

__global__ void Pre_GPU_ScalarProdPart2(double* ts_d, double* vector_d, int subseqLen, int batchIterations, int subseqNum) {

    for (int i = 0; i < batchIterations; i++) {
        int index = (threadIdx.x + blockIdx.x * blockDim.x) + (i * gridDim.x * blockDim.x) + 1;
        if (index < subseqNum) {
            double sum;
            int y = 1;
            for (int x = index % subseqNum; x < subseqNum; x++) {
                if (x > y) {
                    sum = vector_d[x - 1 + (y - 1) * subseqNum] - ts_d[x - 1] * ts_d[y - 1] + ts_d[x + subseqLen - 1] * ts_d[y + subseqLen - 1];
                    vector_d[x + (y * subseqNum)] = sum;
                    vector_d[y + (x * subseqNum)] = sum;
                    y++;
                }
            }
        }
    }
}

__global__ void Pre_GPU_ScalarProdMergePart1(double* ts_d, double* ts2_d, double* vector_d, int subseqLen, int batchIterations, int subseqNum) {

    for (int i = 0; i < batchIterations; i++) {
        int index = (threadIdx.x + blockIdx.x * blockDim.x) + (i * gridDim.x * blockDim.x);
        if (index < subseqNum) {
            int x = index % subseqNum;
            int y = index / subseqNum;
            double sum = 0;
            double sum2 = 0;
            for (int i = 0; i < subseqLen; i++) {
                sum += ts_d[x + i] * ts2_d[y + i];
                sum2 += ts_d[y + i] * ts2_d[x + i];
            }
            vector_d[x + (y * subseqNum)] = sum;
            vector_d[y + (x * subseqNum)] = sum2;
        }
    }
}

__global__ void Pre_GPU_ScalarProdMergePart2(double* ts_d, double* ts2_d, double* vector_d, int subseqLen, int batchIterations, int subseqNum) {

    for (int i = 0; i < batchIterations; i++) {
        int index = (threadIdx.x + blockIdx.x * blockDim.x) + (i * gridDim.x * blockDim.x) + 1;
        if (index < subseqNum) {
            double sum;
            int y = 1;
            for (int x = index % subseqNum; x < subseqNum; x++) {
                int x_1 = x - 1;
                int y_1 = y - 1;
                if (x > y) {
                    sum = vector_d[x_1 + (y_1) * subseqNum] - ts_d[x_1] * ts2_d[y_1] + ts_d[x_1 + subseqLen] * ts2_d[y_1 + subseqLen];

                    vector_d[x + (y * subseqNum)] = sum;
                    y++;
                }
            }

        }
        else if (index < subseqNum + subseqNum) {
            index -= subseqNum;
            double sum2;
            int y = 1;
            for (int x = index % subseqNum; x < subseqNum; x++) {
                int x_1 = x - 1;
                int y_1 = y - 1;
                if (x > y) {
                    sum2 = vector_d[y_1 + (x_1)*subseqNum] - ts_d[y_1] * ts2_d[x_1] + ts_d[y_1 + subseqLen] * ts2_d[x_1 + subseqLen];

                    vector_d[y + (x * subseqNum)] = sum2;
                    y++;
                }
            }
        }
    }
}

__global__ void Pre_GPU_ED(double* vector_d, double* mu_d, double* sigma_d, int subseqLen, int batchIterations, int subseqNum) {

    for (int i = 0; i < batchIterations; i++) {

        int index = (threadIdx.x + blockIdx.x * blockDim.x) + (i * gridDim.x * blockDim.x);

        if (index < subseqNum * subseqNum) {
            int x = index % subseqNum;
            int y = index / subseqNum;

            if (x != y) {

                double ed = sqrt(2 * subseqLen * (1 - ((vector_d[index] - (subseqLen * mu_d[x] * mu_d[y])) / (subseqLen * sigma_d[x] * sigma_d[y]))));

                vector_d[index] = ed;
            }
            else {
                vector_d[index] = INFINITY;
            }
        }
    }
}

__global__ void Pre_GPU_EDMerge(double* vector_d, double* mu1_d, double* sigma1_d, double* mu2_d, double* sigma2_d, int subseqLen, int batchIterations, int subseqNum) {

    for (int i = 0; i < batchIterations; i++) {

        int index = (threadIdx.x + blockIdx.x * blockDim.x) + (i * gridDim.x * blockDim.x);

        if (index < subseqNum * subseqNum) {
            int x = index % subseqNum;
            int y = index / subseqNum;

            //double ed = (2 * subseqLen * (1 - ((scalar_d[index] - (subseqLen * mu_d[x] * mu2_d[y])) / (subseqLen * sigma_d[x] * sigma2_d[y]))));

            double ed = __fsqrt_rn(2 * subseqLen * (1 - ((vector_d[index] - (subseqLen * mu1_d[x] * mu2_d[y])) / (subseqLen * sigma1_d[x] * sigma2_d[y]))));
            //double ed = __fsqrt_rn(2 * subseqLen * (1 - ((scalar_d[index] - (__fmul_rn(__fmul_rn(subseqLen, mu_d[x]), mu2_d[y]))) / (__fmul_rn(__fmul_rn(subseqLen,sigma_d[x]), sigma2_d[y])))));

            vector_d[index] = ed;
        }
    }
}

__global__ void Pre_GPU_MP(double* vector_d, double* mp_d, int* mpIndex_d, int BatchIterations, int subseqNum) {

    for (int i = 0; i < BatchIterations; i++) {
        
        int index = (threadIdx.x + blockIdx.x * blockDim.x) + (i * gridDim.x * blockDim.x);

        if (index < subseqNum) {
            int mpIndex = 0;
            int mpMin = 0;
            double mp = INFINITY;

            int index_x = 0;
            for (int j = 0; j < subseqNum; j++) {
                index_x = j + index * subseqNum;
                if (vector_d[index_x] < mp) {
                    mp = vector_d[index_x];
                    mpIndex = index_x;
                }
            }
            mp_d[index] = mp;
            mpIndex_d[index] = mp;
        }
    }
}

__global__ void Pre_GPU_MPMerge(double* mp1_d, int* mpIndex1_d, double* mp2_d, int* mpIndex2_d, double* vector_d, int subseqNum, int batchIterations) {

    for (int i = 0; i < batchIterations; i++) {

        int index = (threadIdx.x + blockIdx.x * blockDim.x) + (i * gridDim.x * blockDim.x);

        if (index < subseqNum) {
            for (int i = 0; i < subseqNum; i++) {
                if (mp1_d[index] > vector_d[index + i * subseqNum]) {
                    mp1_d[index] = vector_d[index + i * subseqNum];
                    mpIndex1_d[index] = i;
                }
                if (mp2_d[index] > vector_d[i + index * subseqNum]) {
                    mp2_d[index] = vector_d[i + index * subseqNum];
                    mpIndex2_d[index] = i;
                }
            }
        }
    }
}

extern "C" __declspec(dllexport) void cudaRun_All(double* ts, int subseqLen, double* mu, double* sigma, double* mp, int* mpIndex) {

    int sizeSubseq = (_msize(ts) / sizeof(double)) - subseqLen + 1;
    int block = 800000;
    //int block = 1024;
    int thread = 1024;
    int allTread = block * thread;

    int batchIterationsMuSigma = sizeSubseq / allTread + 1;

    int batchIterationsScalED = (sizeSubseq * sizeSubseq) / allTread + 1;

    int batchIterationsMP = (sizeSubseq - 1) / (allTread)+1;

    double* ts_d;
    double* mu_d;
    double* sigma_d;
    double* vector_d;
    double* mp_d;
    int* mpIndex_d;

    cudaMalloc((void**)&ts_d, _msize(ts));
    cudaMalloc((void**)&mu_d, _msize(mu));
    cudaMalloc((void**)&sigma_d, _msize(sigma));
    cudaMalloc((void**)&vector_d, sizeSubseq * sizeSubseq * sizeof(double));
    cudaMalloc((void**)&mp_d, _msize(mp));
    cudaMalloc((void**)&mpIndex_d, _msize(mpIndex));

    cudaMemcpy(ts_d, ts, _msize(ts), cudaMemcpyKind::cudaMemcpyHostToDevice);

    float time;

    cudaEvent_t start, stop;

    // ---------- MuSigma -----------
    
    Pre_GPU_MuSigma << <block, thread >> > (ts_d, mu_d, sigma_d, subseqLen, batchIterationsMuSigma, sizeSubseq);

    // ---------- Scal -----------

    Pre_GPU_ScalarProdPart1 << <block, thread >> > (ts_d, vector_d, subseqLen, batchIterationsScalED, sizeSubseq);

    // ---------- Scal -----------

    Pre_GPU_ScalarProdPart2 << <block, thread >> > (ts_d, vector_d, subseqLen, batchIterationsScalED, sizeSubseq);

    // ---------- ED -----------

    Pre_GPU_ED << <block, thread >> > (vector_d, mu_d, sigma_d, subseqLen, batchIterationsScalED, sizeSubseq);

    // ---------- MP -----------

    Pre_GPU_MP << <block, thread >> > (vector_d, mp_d, mpIndex_d, batchIterationsMP, sizeSubseq);

    cudaMemcpy(mp, mp_d, _msize(mp), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaMemcpy(mu, mu_d, _msize(mp), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaMemcpy(sigma, sigma_d, _msize(mp), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaMemcpy(mpIndex, mpIndex_d, _msize(mpIndex), cudaMemcpyKind::cudaMemcpyDeviceToHost);

    cudaFree(ts_d);
    cudaFree(mu_d);
    cudaFree(sigma_d);
    cudaFree(vector_d);
    cudaFree(mp_d);
    cudaFree(mpIndex_d);
}

extern "C" __declspec(dllexport) void cudaRun_Union(int subseqLen, double* ts, double* mu, double* sigma, double* mp, int* mpIndex, double* ts2, double* mu2, double* sigma2, double* mp2, int* mpIndex2) {

    int sizeSubseq = (_msize(ts) / sizeof(double)) - subseqLen + 1;
    int block = 800000;
    int thread = 1024;
    int allTread = block * thread;

    //int moreMuSigma = sizeSubseq / allTread + 1;

    int batchIterationsScalED = (sizeSubseq * sizeSubseq) / allTread + 1;

    int batchIterationsMP = (sizeSubseq - 1) / (allTread)+1;

    double* ts_d;
    double* ts2_d;

    double* mu_d;
    double* sigma_d;
    double* mu2_d;
    double* sigma2_d;

    double* vector_d;

    double* mp_d;
    double* mp2_d;
    double* mpUnion_d;
    int* mpIndex_d;
    int* mpIndex2_d;
    int* mpIndexUnion_d;

    cudaMalloc((void**)&ts_d, _msize(ts));
    cudaMalloc((void**)&ts2_d, _msize(ts));

    cudaMalloc((void**)&mu_d, _msize(mu));
    cudaMalloc((void**)&sigma_d, _msize(sigma));
    cudaMalloc((void**)&mu2_d, _msize(mu));
    cudaMalloc((void**)&sigma2_d, _msize(sigma));

    cudaMalloc((void**)&vector_d, sizeSubseq * sizeSubseq * sizeof(double));

    cudaMalloc((void**)&mp_d, _msize(mp));
    cudaMalloc((void**)&mp2_d, _msize(mp));
    cudaMalloc((void**)&mpUnion_d, _msize(mp));

    cudaMalloc((void**)&mpIndex_d, _msize(mpIndex));
    cudaMalloc((void**)&mpIndex2_d, _msize(mpIndex));
    cudaMalloc((void**)&mpIndexUnion_d, _msize(mpIndex));

    cudaMemcpy(ts_d, ts, _msize(ts), cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(ts2_d, ts2, _msize(ts), cudaMemcpyKind::cudaMemcpyHostToDevice);

    cudaMemcpy(mu_d, mu, _msize(mu), cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(mu2_d, mu2, _msize(mu), cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(sigma_d, sigma, _msize(sigma), cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(sigma2_d, sigma2, _msize(sigma), cudaMemcpyKind::cudaMemcpyHostToDevice);

    cudaMemcpy(mp_d, mp, _msize(mp), cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(mp2_d, mp2, _msize(mp), cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(mpIndex_d, mpIndex, _msize(mpIndex), cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(mpIndex2_d, mpIndex2, _msize(mpIndex), cudaMemcpyKind::cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float time;

    // ---------- ScalMerg -----------

    Pre_GPU_ScalarProdMergePart1 << <block, thread >> > (ts_d, ts2_d, vector_d, subseqLen, batchIterationsScalED, sizeSubseq);

    // ---------- ScalMerg -----------

    Pre_GPU_ScalarProdMergePart2 << <block, thread >> > (ts_d, ts2_d, vector_d, subseqLen, batchIterationsScalED, sizeSubseq);

    // ---------- EDMerg -----------

    Pre_GPU_EDMerge << <block, thread >> > (vector_d, mu_d, sigma_d, mu2_d, sigma2_d, subseqLen, batchIterationsScalED, sizeSubseq);

    // ---------- MPMerg -----------

    Pre_GPU_MPMerge << <block, thread >> > (mp_d, mpIndex_d, mp2_d, mpIndex2_d, vector_d, sizeSubseq, batchIterationsMP);

    cudaMemcpy(mp, mp_d, _msize(mp), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaMemcpy(mpIndex, mpIndex_d, _msize(mpIndex), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaMemcpy(mp2, mp2_d, _msize(mp), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaMemcpy(mpIndex2, mpIndex2_d, _msize(mpIndex), cudaMemcpyKind::cudaMemcpyDeviceToHost);

    cudaFree(ts_d);
    cudaFree(ts2_d);
    cudaFree(mu_d);
    cudaFree(sigma_d);
    cudaFree(mu2_d);
    cudaFree(sigma2_d);

    cudaFree(vector_d);

    cudaFree(mp_d);
    cudaFree(mp2_d);
    cudaFree(mpUnion_d);
    
    cudaFree(mpIndex_d);
    cudaFree(mpIndex2_d);
    cudaFree(mpIndexUnion_d);
}
