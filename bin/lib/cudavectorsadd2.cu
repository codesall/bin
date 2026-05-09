!nvidia-smi
!nvcc --version
!pip install nvcc4jupyter
%load_ext nvcc4jupyter


%%cuda
#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void vectorAdd(int* a, int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    int n = 1000000;
    size_t size = n * sizeof(int);

    int *h_a = new int[n], *h_b = new int[n], *h_c = new int[n];
    for (int i = 0; i < n; i++) { h_a[i] = i; h_b[i] = i * 2; }

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    cout << "Vector size : " << n << endl;
    cout << "Threads/block: " << threads << " | Blocks: " << blocks << endl;
    cout << "GPU Time     : " << ms << " ms" << endl;
    cout << "Sample: a[0]=" << h_a[0] << " b[0]=" << h_b[0] << " c[0]=" << h_c[0] << endl;
    cout << "Sample: a[1]=" << h_a[1] << " b[1]=" << h_b[1] << " c[1]=" << h_c[1] << endl;
    cout << "Sample: a[9]=" << h_a[9] << " b[9]=" << h_b[9] << " c[9]=" << h_c[9] << endl;

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] h_a; delete[] h_b; delete[] h_c;
    return 0;
}

