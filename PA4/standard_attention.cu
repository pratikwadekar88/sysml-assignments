// Standard (non-flash) attention: materializes the full N x N score matrix
// in global memory, then does row-wise softmax, then multiplies by V.
// Mirrors flash.cu's problem setup (B, NH, N, d) and CLI/timing conventions
// so the two can be benchmarked apples-to-apples.
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

// S[i][j] = scale * Q[i] . K[j],  one thread per (i,j) output element.
__global__ void compute_scores(const float* Q, const float* K, float* S,
                                int N, int d, float scale) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N) {
        float dot = 0.0f;
        for (int x = 0; x < d; x++) dot += Q[i * d + x] * K[j * d + x];
        S[i * N + j] = dot * scale;
    }
}

// In-place row softmax: one block per row, block-wide max/sum reductions.
__global__ void softmax_rows(float* S, int N) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    float* row_ptr = S + (size_t)row * N;

    float local_max = -1e20f;
    for (int j = tid; j < N; j += nthreads) local_max = fmaxf(local_max, row_ptr[j]);
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < N; j += nthreads) {
        float v = expf(row_ptr[j] - row_max);
        row_ptr[j] = v;
        local_sum += v;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();

    for (int j = tid; j < N; j += nthreads) row_ptr[j] /= row_sum;
}

// O[i][x] = sum_j S[i][j] * V[j][x],  one thread per (i,x) output element.
__global__ void compute_output(const float* S, const float* V, float* O,
                                int N, int d) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && x < d) {
        float acc = 0.0f;
        for (int j = 0; j < N; j++) acc += S[i * N + j] * V[j * d + x];
        O[i * d + x] = acc;
    }
}

int main(int argc, char *argv[]) {
    int N = 1024;
    if (argc >= 2) N = atoi(argv[1]);

    int B = 1, NH = 1, d = 64;
    float scale = 1.0f / sqrtf((float)d);

    size_t qkv_size = (size_t)B * NH * N * d * sizeof(float);
    size_t s_size = (size_t)B * NH * N * N * sizeof(float);

    float *h_Q, *h_K, *h_V, *h_O;
    cudaMallocHost(&h_Q, qkv_size);
    cudaMallocHost(&h_K, qkv_size);
    cudaMallocHost(&h_V, qkv_size);
    cudaMallocHost(&h_O, qkv_size);
    for (int i = 0; i < B * NH * N * d; ++i) {
        h_Q[i] = 0.1f;
        h_K[i] = 0.1f;
        h_V[i] = 0.1f;
    }

    float *d_Q, *d_K, *d_V, *d_O, *d_S;
    cudaMalloc(&d_Q, qkv_size);
    cudaMalloc(&d_K, qkv_size);
    cudaMalloc(&d_V, qkv_size);
    cudaMalloc(&d_O, qkv_size);
    cudaMalloc(&d_S, s_size);

    cudaMemcpy(d_Q, h_Q, qkv_size, cudaMemcpyDefault);
    cudaMemcpy(d_K, h_K, qkv_size, cudaMemcpyDefault);
    cudaMemcpy(d_V, h_V, qkv_size, cudaMemcpyDefault);

    dim3 block2d(16, 16);
    dim3 grid_scores((N + 15) / 16, (N + 15) / 16);
    dim3 grid_out((d + 15) / 16, (N + 15) / 16);
    int softmax_threads = 256;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int b = 0; b < B * NH; b++) {
        compute_scores<<<grid_scores, block2d>>>(d_Q + b * N * d, d_K + b * N * d, d_S + (size_t)b * N * N, N, d, scale);
        softmax_rows<<<N, softmax_threads, softmax_threads * sizeof(float)>>>(d_S + (size_t)b * N * N, N);
        compute_output<<<grid_out, block2d>>>(d_S + (size_t)b * N * N, d_V + b * N * d, d_O + b * N * d, N, d);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_O, d_O, qkv_size, cudaMemcpyDefault);
    cout << "StandardAttention Kernel Time: " << milliseconds / 1000.0f << " s" << endl;
    cout << "StandardAttention Score Matrix Memory: " << s_size << " bytes" << endl;

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_S);
    cudaFreeHost(h_Q); cudaFreeHost(h_K); cudaFreeHost(h_V); cudaFreeHost(h_O);

    return 0;
}
