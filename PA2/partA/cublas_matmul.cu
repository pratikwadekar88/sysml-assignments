#include <iostream>
#include <random>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
using namespace std;


void matmul_reference(float* A, float* B, float* C, int N)
{
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            float sum = 0.0f;
            for(int k=0;k<N;k++){
                sum += A[i*N+k]*B[k*N+j];
            }
            C[i*N+j] = sum;
        }
    }
}

/* Compare matrices */
bool verify(float* C1, float* C2, int N)
{
    const float eps = 1e-3;

    for(int i = 0; i < N*N; i++){
        if (fabs(C1[i] - C2[i]) > eps){
            cout << "Mismatch at index " << i
                 << " : " << C1[i] << " vs " << C2[i] << endl;
            return false;
        }
    }
    return true;
}

void initialize_matrix(float* M, int N)
{
    static random_device rd;
    static mt19937 gen(rd());
    static uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (long long i = 0; i < (long long)N * N; i++)
        M[i] = dist(gen);
}

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        cout << "Usage: ./benchmark_gpu <N>\n";
        return 1;
    }

    int N = atoi(argv[1]);

    size_t bytes = (size_t)N * N * sizeof(float);

    float *A, *B, *C,*C_ref;


    cudaMallocHost(&A,bytes);
    cudaMallocHost(&B,bytes);
    cudaMallocHost(&C,bytes);
    cudaMallocHost(&C_ref, bytes);
    initialize_matrix(A, N);
    initialize_matrix(B, N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A,bytes);
    cudaMalloc(&d_B,bytes);
    cudaMalloc(&d_C,bytes);
    

    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);


    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    // cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,&alpha,d_B,N,d_A,N,&beta,d_C,N);
    cublasSgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_T,
            N, N, N,
            &alpha,
            d_A, N,
            d_B, N,
            &beta,
            d_C, N);

    cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);
    
    matmul_reference(A, B, C_ref, N);

        if (verify(C, C_ref, N))
            cout << "Verification PASSED\n";
        else
            cout << "Verification FAILED\n";

    cout << "C[0] = " << C[0] << endl;

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);

    return 0;
}