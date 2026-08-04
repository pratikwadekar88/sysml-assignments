#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <stdio.h>
// #include <cuda/cmath>
#include<iostream>
using namespace std;

__global__ void vecAdd(float *A, float *B,float *C,int N){

    int i = threadIdx.x + blockIdx.x *blockDim.x;
    if(i<N) C[i] = A[i] + B[i];

}

void initArray(float *A, int N){
    for(int i=0;i<N;i++){
        A[i] = rand()%10 +1;
    }
}


int main(){
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;

    int N = 2048;
    cudaMallocManaged(&A,N*sizeof(float));
    cudaMallocManaged(&B,N*sizeof(float));
    cudaMallocManaged(&C,N*sizeof(float));

    initArray(A,N);
    initArray(B,N);

    int threads = 256;

    // int blocks = cuda::ceil_div(N,threads);
    int blocks = (N+ threads-1)/threads;

    vecAdd<<<blocks,threads>>>(A,B,C,N);

    cudaDeviceSynchronize();

    for(int i=0;i<N;i++){
        cout<<A[i]<<" + "<<B[i]<<" = "<<C[i]<<endl;
    }
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
    
}