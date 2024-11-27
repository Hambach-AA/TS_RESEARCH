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
__global__ void Scalar(float* expansionTS_d, int* scalar_d, int sizeMat, int subseqLen, int more) {

    for (int i = 0; i < more; i++) {
        int index_x = (threadIdx.x + blockIdx.x * blockDim.x) + (i * gridDim.x * blockDim.x);
        for (int j = 0; j < more; j++) {
            int index_y = (threadIdx.y + blockIdx.y * blockDim.y) + (j * gridDim.y * blockDim.y);
            int sum;
            if (index_x != index_y && index_x > index_y) {
                sum = 0;
                int iy = index_y;
                for (int ix = index_x; ix < index_x + subseqLen; ix++) {
                    sum += expansionTS_d[ix] * expansionTS_d[iy];
                    iy++;
                }
                scalar_d[index_x + sizeMat * index_y] = sum;
            }
        }
    }
}


void cudaRan_MuSigma(float * ts, int sizeTS, int subseqLen, char * out_mu, char * out_sigma) {

    int sizeSubseq = sizeTS - subseqLen + 1;
    int block = 100;
    int thread = 1000;
    int allTread = block * thread;
    int more = sizeSubseq / allTread + 1;
    int num = more * allTread;

    float *expansionTS = new float[num + subseqLen - 1] {} ;
    float *mu = new float[num] {} ;
    float *sigma = new float[num] {} ;
    
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

    std::ofstream file;
    file.open(out_mu);
    for (int i = 0; i < sizeSubseq; i++) {
        file << std::to_string(mu[i]) << " " << i << "\n";
    }
    file.close();

    file.open(out_sigma);
    for (int i = 0; i < sizeSubseq; i++) {
        file << std::to_string(sigma[i]) << " " << i << "\n";
    }
    file.close();
}


void cudaRan_Scalar(float* ts, int sizeTS, int subseqLen, char * out_scalar) {

    int sizeSubseq = sizeTS - subseqLen + 1;
    int block_n = 100;
    int thread_n = 10;
    int allTread = thread_n * block_n;

    dim3 block = dim3(block_n, block_n);
    dim3 thread = dim3(thread_n, thread_n);

    int more = sizeSubseq / allTread + 1;

    int num = more * allTread;

    float* expansionTS = new float[num + subseqLen - 1];
    for (int i = 0; i < sizeTS; i++) {
        expansionTS[i] = ts[i];
    }

    int* scalar = new int[num * num] {};

    float* expansionTS_d;
    int* scalar_d;

    cudaMalloc((void**)&expansionTS_d, sizeof(float) * (num + subseqLen - 1));
    cudaMalloc((void**)&scalar_d, sizeof(int) * num * num);


    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMemcpy(expansionTS_d, expansionTS, sizeof(float) * (num + subseqLen - 1), cudaMemcpyKind::cudaMemcpyHostToDevice);

    Scalar <<<block, thread >>> (expansionTS_d, scalar_d, num, subseqLen, more);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time work: %f\n", time);

    cudaMemcpy(scalar, scalar_d, sizeof(int) * num * num, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    cudaFree(expansionTS_d);
    cudaFree(scalar_d);

    std::ofstream file;
    file.open(out_scalar);
    for (int i = 0; i < sizeSubseq; i++) {
        for (int j = 0; j < sizeSubseq; j++) {
            if (i != j && i > j) {
                file << std::to_string(scalar[i + num * j]) << " " << i << " " << j << "\n";
            }
        }
    }
    file.close();
}

int main(int argc, char** argv)
{

    float* ts = new float[std::atoi(argv[2])];
    std::ifstream file;
    float num;
    file.open(argv[1]);
    int i = 0;
    while (file >> num) {
        ts[i++] = num;
    }
    file.close();

    int subseqLen = std::atoi(argv[3]);

    int sizeTS = _msize(ts) / sizeof(float);

    cudaRan_MuSigma(ts, sizeTS, subseqLen, argv[4], argv[5]);
    cudaRan_Scalar(ts, sizeTS, subseqLen, argv[6]);
    

    return 0;
}
