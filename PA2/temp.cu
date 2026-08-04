#include <iostream>
#include <random>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

#define TILE 16
#define NELEM 4

/* ===========================
   CUDA KERNEL
   =========================== */

__global__
void matmul_kernel(float* A, float* B, float* C, int N)
{
    extern __shared__ float s[];

    float *sA = s;
    float *sB = s + TILE*TILE;

    int localRow = threadIdx.y;
    int localCol = threadIdx.x;

    int globalRow = blockIdx.y * TILE + localRow;
    int globalCol = blockIdx.x * (TILE*NELEM) + localCol;

    float acc[NELEM];

    for(int i=0;i<NELEM;i++)
        acc[i] = 0.0f;

    int numTile = (N + TILE - 1) / TILE;

    for(int t=0; t<numTile; t++)
    {

        /* ---- Load A tile ---- */

        int aRow = globalRow;
        int aCol = t*TILE + localCol;

        if(aRow < N && aCol < N)
            sA[localRow*TILE + localCol] = A[aRow*N + aCol];
        else
            sA[localRow*TILE + localCol] = 0.0f;


        /* ---- Load B tile (strided) ---- */

        for(int i=0;i<NELEM;i++)
        {
            int bRow = t*TILE + localRow;
            int bCol = globalCol + i*TILE;

            if(bRow < N && bCol < N)
                sB[localRow*(TILE*NELEM) + localCol + i*TILE] =
                    B[bRow*N + bCol];
            else
                sB[localRow*(TILE*NELEM) + localCol + i*TILE] = 0.0f;
        }

        __syncthreads();


        /* ---- Compute ---- */

        for(int k=0;k<TILE;k++)
        {
            float a = sA[localRow*TILE + k];

            for(int i=0;i<NELEM;i++)
            {
                acc[i] += a *
                    sB[k*(TILE*NELEM) + localCol + i*TILE];
            }
        }

        __syncthreads();
    }


    /* ---- Store results ---- */

    for(int i=0;i<NELEM;i++)
    {
        int col = globalCol + i*TILE;

        if(globalRow < N && col < N)
            C[globalRow*N + col] = acc[i];
    }
}

/* ===========================
   CPU Reference
   =========================== */

void matmul_reference(float* A, float* B, float* C, int N)
{
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
        {
            float sum = 0;

            for(int k=0;k<N;k++)
                sum += A[i*N+k] * B[k*N+j];

            C[i*N+j] = sum;
        }
}

/* ===========================
   VERIFY
   =========================== */

bool verify(float* C1, float* C2, int N)
{
    const float eps = 1e-3;

    for(int i=0;i<N*N;i++)
    {
        if(fabs(C1[i] - C2[i]) > eps)
        {
            cout<<"Mismatch "<<C1[i]<<" "<<C2[i]<<endl;
            return false;
        }
    }
    return true;
}

/* ===========================
   INIT
   =========================== */

void initialize_matrix(float* M,int N)
{
    for(long long i=0;i<(long long)N*N;i++)
        M[i] = rand()/(float)RAND_MAX;
}

/* ===========================
   MAIN
   =========================== */

int main(int argc,char* argv[])
{
    int N = atoi(argv[1]);

    size_t bytes = N*N*sizeof(float);

    float *A,*B,*C,*C_ref;

    cudaMallocHost(&A,bytes);
    cudaMallocHost(&B,bytes);
    cudaMallocHost(&C,bytes);
    cudaMallocHost(&C_ref,bytes);

    initialize_matrix(A,N);
    initialize_matrix(B,N);

    float *d_A,*d_B,*d_C;

    cudaMalloc(&d_A,bytes);
    cudaMalloc(&d_B,bytes);
    cudaMalloc(&d_C,bytes);

    cudaMemcpy(d_A,A,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,bytes,cudaMemcpyHostToDevice);


    int sharedMemSize =
        (TILE*TILE + TILE*TILE*NELEM)*sizeof(float);

    dim3 block(TILE,TILE);

    dim3 grid(
        (N + TILE*NELEM - 1)/(TILE*NELEM),
        (N + TILE - 1)/TILE
    );


    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matmul_kernel<<<grid,block,sharedMemSize>>>(
        d_A,d_B,d_C,N
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms=0;
    cudaEventElapsedTime(&ms,start,stop);

    cudaMemcpy(C,d_C,bytes,cudaMemcpyDeviceToHost);


    matmul_reference(A,B,C_ref,N);

    if(verify(C,C_ref,N))
        cout<<"Verification PASSED\n";
    else
        cout<<"Verification FAILED\n";

    cout<<"Kernel Time (s): "<<ms/1000.0f<<endl;


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    cudaFreeHost(C_ref);
}