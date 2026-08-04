#include <iostream>
#include <random>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

/* ===========================
   WRITE YOUR KERNEL HERE
   =========================== */

__global__
void matmul_kernel(float* A, float* B, float* C, int N,int TILE)
{
    // WRITE CUDA MATRIX MULTIPLICATION HERE
    extern __shared__ float s[];
    float *sA = s;
    float *sB = s + TILE*TILE;

    int globalRow = blockIdx.y* blockDim.y + threadIdx.y;
    int globalCol = blockIdx.x* blockDim.x + threadIdx.x;

    int localRow = threadIdx.y;
    int localCol = threadIdx.x;

    float acc = 0.0f;

    int numTile = (N+TILE-1)/TILE;

    for(int t=0;t<numTile;t++){
        int aRow = globalRow;
        int aCol = t*TILE+localCol;
        int bRow = t*TILE + localRow;
        int bCol = globalCol;

        if(aRow<N and aCol<N) sA[localRow*TILE+localCol] = A[aRow*N+aCol];
        else sA[localRow*TILE+localCol] = 0.0f;

        if(bRow<N and bCol<N) sB[localRow*TILE+ localCol] = B[bRow*N+bCol];
        else  sB[localRow*TILE+ localCol] = 0.0f;

        __syncthreads();

        for(int k=0;k<TILE;k++){
            acc+= sA[localRow*TILE+k]*sB[k*TILE+localCol];
        }
        __syncthreads();

    }
    if (globalRow < N && globalCol < N)
        C[globalRow*N+globalCol] = acc;

    
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
    // cudaMallocHost(&C_ref, bytes);

    initialize_matrix(A, N);
    initialize_matrix(B, N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A,bytes);
    cudaMalloc(&d_B,bytes);
    cudaMalloc(&d_C,bytes);
    

    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    int TILE = 32;
    int sharedMemSize = 2 * TILE * TILE * sizeof(float);
    dim3 block(32,32);
    dim3 grid((N+31)/32, (N+31)/32);

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_kernel<<<grid, block,sharedMemSize>>>(d_A, d_B, d_C, N,TILE);
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