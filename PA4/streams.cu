#include <iostream>
#include <random>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

__global__ void naive_matmul(float* A,float *B,float *C,int N, int b ){

    int i = blockIdx.y* blockDim.y + threadIdx.y;
    int j = blockIdx.x* blockDim.x + threadIdx.x;

    float acc = 0.0f;
    for(int k=0;k<N;k++){
        if(i<b and j<N)
            acc += A[i*N+k] * B[k*N+j]; 
    }
    if(i<b and j<N) C[i*N+j] = acc;

}

__global__ void relu(float* Y,int N,int B){
    int i = blockIdx.y* blockDim.y + threadIdx.y;
    int j = blockIdx.x* blockDim.x + threadIdx.x;

    // Y[i*N+j] = max(0,Y[i*N+j]);
    if (i<B and j<N and Y[i*N+j]<0) Y[i*N+j] =0;


}

void initialize_matrix(float* M, int N,int B)
{
    static random_device rd;
    static mt19937 gen(rd());
    static uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (long long i = 0; i < (long long)N * B; i++)
        M[i] = 1;
}

int main(){

    int B = 32;
    int N = 2048;

    float *X,*Y,*Z,*W1,*W2;

    int ws = N*N;
    int as = B*N;

    cudaMallocHost(&X,as*sizeof(float));
    cudaMallocHost(&Z,as*sizeof(float));
    cudaMallocHost(&W1,ws*sizeof(float));
    cudaMallocHost(&W2,ws*sizeof(float));

    initialize_matrix(X,B,N);
    initialize_matrix(W1,N,N);
    initialize_matrix(W2,N,N);

    float *dX,*dY,*dZ,*dW1,*dW2;

    cudaMalloc(&dX,as*sizeof(float));
    cudaMalloc(&dY,as*sizeof(float));
    cudaMalloc(&dW1,ws*sizeof(float));
    cudaMalloc(&dW2,ws*sizeof(float));
    cudaMalloc(&dZ,as*sizeof(float)); 

    cudaMemcpy(dW1,W1,ws*sizeof(float),cudaMemcpyDefault);
    cudaMemcpy(dW2,W2,ws*sizeof(float),cudaMemcpyDefault);

    int blksz = 4;
    int ns = 4;
    cudaStream_t s[ns];
    for(int i=0;i<ns;i++) cudaStreamCreate(&s[i]);

    int rps = B/ns;
    int eps = (B*N+ns-1)/ns;

    dim3 block(blksz,blksz);
    dim3 grid(
        (N+blksz-1)/blksz,
        (rps+blksz-1)/blksz
    );

    for(int i=0;i<ns;i++){
        float *dX_off = dX + i*eps;
        float *dZ_off = dZ + i*eps;
        float *dY_off = dY+ i*eps;

        float *X_off = X + i*eps;
        float *Z_off = Z + i*eps;

        cudaMemcpyAsync(dX_off,X_off,eps*sizeof(float),cudaMemcpyDefault,s[i]);
        naive_matmul<<<grid,block,0,s[i]>>>(dX_off,dW1,dY_off,N,rps);
        relu<<<grid,block,0,s[i]>>>(dY_off,N,rps);
        naive_matmul<<<grid,block,0,s[i]>>>(dY_off,dW2,dZ_off,N,rps);
        cudaMemcpyAsync(Z_off,dZ_off,eps*sizeof(float),cudaMemcpyDefault,s[i]);
    }

    cudaDeviceSynchronize();
    for(int i=0;i<ns;i++) cudaStreamDestroy(s[i]);

    cout<<Z[0]<<endl;
    // for(int i=0;i<B;i++){
    //     for(int j=0;j<N;j++)
    //         cout<<Z[i*N+j]<<" ";
    //     cout<<endl;
    // }

    return 0;
}

