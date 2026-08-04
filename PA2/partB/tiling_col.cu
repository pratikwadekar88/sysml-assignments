#include <iostream>
#include <random>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

/* ===========================
   WRITE YOUR KERNEL HERE
   =========================== */
#define NELEM 16
#define TILE 16

__global__
void matmul_kernel(float* A, float* B, float* C, int N,int numTile)
{
    extern __shared__ float s[];
    float *sA = s;
    float *sB = s+ (TILE*NELEM) * TILE;

    int localRow = threadIdx.y;
    int localCol = threadIdx.x;
    int globalRow = blockIdx.y* (NELEM*TILE) + localRow;
    int globalCol = blockIdx.x * blockDim.x + localCol;

    float acc[NELEM];
    for(int i=0;i<NELEM;i++)
    acc[i] = 0.0f;
    for(int tB=0;tB<numTile;tB++){

        for(int i=0;i<NELEM;i++){
            int aRow = globalRow + i*TILE;
            int aCol = tB*TILE+localCol; //IMP

            if(aRow<N and aCol<N) sA[i*TILE*TILE + localRow*TILE + localCol] = A[aRow*N+aCol];
            else sA[i*TILE*TILE + localRow*TILE + localCol] = 0.0f;
        }

        int bRow = tB*TILE+localRow; 
        int bCol = globalCol;

        if(bRow<N and bCol<N) sB[localRow*TILE+localCol] = B[bRow*N+bCol];
        else sB[localRow*TILE+localCol]  = 0.0f;

        __syncthreads();

        for(int k=0;k<TILE;k++){
            float b = sB[k*TILE+localCol];
            for(int i=0;i<NELEM;i++){
                acc[i] += sA[i*TILE*TILE + localRow*TILE + k] *b;
            }
        }
        __syncthreads();
    }

    for(int i=0;i<NELEM;i++){
        int cRow = globalRow + i*TILE;
        int cCol = globalCol;
        if(cRow<N and cCol<N) C[cRow*N+cCol] = acc[i];
    }

    
}

/* =========================== */

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

    // int NELEM = 4;

    int TileSizeA = NELEM*TILE;
    int sharedMemSize = (NELEM*TILE*TILE + TILE*TILE) * sizeof(float);
    int numTile = (N+TILE-1)/TILE;

    dim3 block(TILE,TILE);
    dim3 grid(
    (N + TILE - 1) / TILE,
    (N + TileSizeA - 1) / TileSizeA
);

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_kernel<<<grid, block,sharedMemSize>>>(d_A, d_B, d_C, N,numTile);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time = 0.0f;
    cudaEventElapsedTime(&time,start,stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);
    
    // matmul_reference(A, B, C_ref, N);

    //     if (verify(C, C_ref, N))
    //         cout << "Verification PASSED\n";
    //     else
    //         cout << "Verification FAILED\n";
    cout << C[0] << endl;
    cout<<"Time(s): "<<time/1000.0f<<endl;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);

    return 0;
}