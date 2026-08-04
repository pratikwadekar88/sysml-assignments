#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

const int N = 1024;
const int IN = 512;
const int H1 = 2048;
const int H2 = 2048;
const int H3 = 2048;
const int OUT = 512;

const float LR = 0.001f;
const int ITERATIONS = 10;

void init_random_host(float* data, int size, float scale = 0.01f) {
    for (int i = 0; i < size; ++i) {
        float r = (float)rand() / (float)RAND_MAX;
        data[i] = (r * 2.0f - 1.0f) * scale;
    }
}

void init_zeros_host(float* data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = 0.0f;
    }
}

__global__ void matmul_kernel(const float* A, const float* B, float* C, const float* bias, 
                              int M, int N_cols, int K, bool transA, bool transB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N_cols) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a = transA ? A[k * M + row] : A[row * K + k];
            float b = transB ? B[col * K + k] : B[k * N_cols + col];
            sum += a * b;
        }
        if (bias != nullptr && !transA && !transB) {
            sum += bias[col];
        }
        C[row * N_cols + col] = sum;
    }
}

__global__ void relu_kernel(const float* Z, float* A, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        A[i] = Z[i] > 0.0f ? Z[i] : 0.0f;
    }
}

__global__ void relu_grad_kernel(const float* dA, const float* Z, float* dZ, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        dZ[i] = Z[i] > 0.0f ? dA[i] : 0.0f;
    }
}

__global__ void mse_grad_kernel(const float* Z, const float* y, float* dZ, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        dZ[i] = 2.0f * (Z[i] - y[i]) / size;
    }
}

__global__ void bias_grad_kernel(const float* dZ, float* db, int batch_size, int out_features) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < out_features) {
        float sum = 0.0f;
        for (int i = 0; i < batch_size; ++i) {
            sum += dZ[i * out_features + col];
        }
        db[col] = sum;
    }
}

__global__ void sgd_update_kernel(float* param, const float* grad, float lr, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        param[i] -= lr * grad[i];
    }
}

void launch_matmul(const float* A, const float* B, float* C, const float* bias, 
                   int M, int N_cols, int K, bool transA, bool transB) {
    dim3 block(32, 32);
    dim3 grid((N_cols + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matmul_kernel<<<grid, block>>>(A, B, C, bias, M, N_cols, K, transA, transB);
}

int main() {
    srand(42);

    size_t mem_free_initial, mem_total;
    cudaMemGetInfo(&mem_free_initial, &mem_total);

    float *h_W1, *h_W2, *h_W3, *h_W4;
    float *h_b1, *h_b2, *h_b3, *h_b4;
    float *h_X, *h_y;

    cudaMallocHost((void**)&h_W1, IN * H1 * sizeof(float));
    cudaMallocHost((void**)&h_W2, H1 * H2 * sizeof(float));
    cudaMallocHost((void**)&h_W3, H2 * H3 * sizeof(float));
    cudaMallocHost((void**)&h_W4, H3 * OUT * sizeof(float));
    cudaMallocHost((void**)&h_b1, H1 * sizeof(float));
    cudaMallocHost((void**)&h_b2, H2 * sizeof(float));
    cudaMallocHost((void**)&h_b3, H3 * sizeof(float));
    cudaMallocHost((void**)&h_b4, OUT * sizeof(float));
    cudaMallocHost((void**)&h_X, N * IN * sizeof(float));
    cudaMallocHost((void**)&h_y, N * OUT * sizeof(float));

    init_random_host(h_W1, IN * H1);
    init_random_host(h_W2, H1 * H2);
    init_random_host(h_W3, H2 * H3);
    init_random_host(h_W4, H3 * OUT);
    init_zeros_host(h_b1, H1);
    init_zeros_host(h_b2, H2);
    init_zeros_host(h_b3, H3);
    init_zeros_host(h_b4, OUT);
    init_random_host(h_X, N * IN, 1.0f);  
    init_random_host(h_y, N * OUT, 1.0f); 

    float *W1, *W2, *W3, *W4;
    float *b1, *b2, *b3, *b4;
    float *X, *y;

    cudaMalloc(&W1, IN * H1 * sizeof(float));
    cudaMalloc(&W2, H1 * H2 * sizeof(float));
    cudaMalloc(&W3, H2 * H3 * sizeof(float));
    cudaMalloc(&W4, H3 * OUT * sizeof(float));
    cudaMalloc(&b1, H1 * sizeof(float));
    cudaMalloc(&b2, H2 * sizeof(float));
    cudaMalloc(&b3, H3 * sizeof(float));
    cudaMalloc(&b4, OUT * sizeof(float));
    cudaMalloc(&X, N * IN * sizeof(float));
    cudaMalloc(&y, N * OUT * sizeof(float));

    cudaMemcpy(W1, h_W1, IN * H1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(W2, h_W2, H1 * H2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(W3, h_W3, H2 * H3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(W4, h_W4, H3 * OUT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b1, h_b1, H1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b2, h_b2, H2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b3, h_b3, H3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b4, h_b4, OUT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(X, h_X, N * IN * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y, h_y, N * OUT * sizeof(float), cudaMemcpyHostToDevice);

    cudaFreeHost(h_W1); cudaFreeHost(h_W2); cudaFreeHost(h_W3); cudaFreeHost(h_W4);
    cudaFreeHost(h_b1); cudaFreeHost(h_b2); cudaFreeHost(h_b3); cudaFreeHost(h_b4);
    cudaFreeHost(h_X); cudaFreeHost(h_y);

    float *Z1, *A1, *Z2, *A2, *Z3, *A3, *Z4;
    cudaMalloc(&Z1, N * H1 * sizeof(float));
    cudaMalloc(&A1, N * H1 * sizeof(float));
    cudaMalloc(&Z2, N * H2 * sizeof(float));
    cudaMalloc(&A2, N * H2 * sizeof(float));
    cudaMalloc(&Z3, N * H3 * sizeof(float));
    cudaMalloc(&A3, N * H3 * sizeof(float));
    cudaMalloc(&Z4, N * OUT * sizeof(float));

    float *dW1, *db1, *dW2, *db2, *dW3, *db3, *dW4, *db4;
    cudaMalloc(&dW1, IN * H1 * sizeof(float));
    cudaMalloc(&db1, H1 * sizeof(float));
    cudaMalloc(&dW2, H1 * H2 * sizeof(float));
    cudaMalloc(&db2, H2 * sizeof(float));
    cudaMalloc(&dW3, H2 * H3 * sizeof(float));
    cudaMalloc(&db3, H3 * sizeof(float));
    cudaMalloc(&dW4, H3 * OUT * sizeof(float));
    cudaMalloc(&db4, OUT * sizeof(float));

    float *dZ4, *dA3, *dZ3, *dA2, *dZ2, *dA1, *dZ1;
    cudaMalloc(&dZ4, N * OUT * sizeof(float));
    cudaMalloc(&dA3, N * H3 * sizeof(float));
    cudaMalloc(&dZ3, N * H3 * sizeof(float));
    cudaMalloc(&dA2, N * H2 * sizeof(float));
    cudaMalloc(&dZ2, N * H2 * sizeof(float));
    cudaMalloc(&dA1, N * H1 * sizeof(float));
    cudaMalloc(&dZ1, N * H1 * sizeof(float));

    size_t mem_free_after;
    cudaMemGetInfo(&mem_free_after, &mem_total);
    std::cout << "GPU Memory Consumed by allocations: " 
              << (mem_free_initial - mem_free_after) / (1024.0 * 1024.0) << " MB\n";

    cudaEvent_t start_fw, stop_fw, start_bw, stop_bw;
    cudaEventCreate(&start_fw); cudaEventCreate(&stop_fw);
    cudaEventCreate(&start_bw); cudaEventCreate(&stop_bw);

    float total_time_fw = 0.0f;
    float total_time_bw = 0.0f;
    
    int threads = 256;
    
    for (int iter = 0; iter < ITERATIONS; ++iter) {
        
        cudaEventRecord(start_fw);
        
        launch_matmul(X, W1, Z1, b1, N, H1, IN, false, false);
        relu_kernel<<<(N * H1 + threads - 1) / threads, threads>>>(Z1, A1, N * H1);

        launch_matmul(A1, W2, Z2, b2, N, H2, H1, false, false);
        relu_kernel<<<(N * H2 + threads - 1) / threads, threads>>>(Z2, A2, N * H2);

        launch_matmul(A2, W3, Z3, b3, N, H3, H2, false, false);
        relu_kernel<<<(N * H3 + threads - 1) / threads, threads>>>(Z3, A3, N * H3);

        launch_matmul(A3, W4, Z4, b4, N, OUT, H3, false, false);

        cudaEventRecord(stop_fw);
        cudaEventSynchronize(stop_fw);

        cudaEventRecord(start_bw);

        mse_grad_kernel<<<(N * OUT + threads - 1) / threads, threads>>>(Z4, y, dZ4, N * OUT);

        launch_matmul(A3, dZ4, dW4, nullptr, H3, OUT, N, true, false);
        bias_grad_kernel<<<(OUT + threads - 1) / threads, threads>>>(dZ4, db4, N, OUT);
        launch_matmul(dZ4, W4, dA3, nullptr, N, H3, OUT, false, true);

        relu_grad_kernel<<<(N * H3 + threads - 1) / threads, threads>>>(dA3, Z3, dZ3, N * H3);
        launch_matmul(A2, dZ3, dW3, nullptr, H2, H3, N, true, false);
        bias_grad_kernel<<<(H3 + threads - 1) / threads, threads>>>(dZ3, db3, N, H3);
        launch_matmul(dZ3, W3, dA2, nullptr, N, H2, H3, false, true);

        relu_grad_kernel<<<(N * H2 + threads - 1) / threads, threads>>>(dA2, Z2, dZ2, N * H2);
        launch_matmul(A1, dZ2, dW2, nullptr, H1, H2, N, true, false);
        bias_grad_kernel<<<(H2 + threads - 1) / threads, threads>>>(dZ2, db2, N, H2);
        launch_matmul(dZ2, W2, dA1, nullptr, N, H1, H2, false, true);

        relu_grad_kernel<<<(N * H1 + threads - 1) / threads, threads>>>(dA1, Z1, dZ1, N * H1);
        launch_matmul(X, dZ1, dW1, nullptr, IN, H1, N, true, false);
        bias_grad_kernel<<<(H1 + threads - 1) / threads, threads>>>(dZ1, db1, N, H1);

        sgd_update_kernel<<<(IN * H1 + threads - 1) / threads, threads>>>(W1, dW1, LR, IN * H1);
        sgd_update_kernel<<<(H1 * H2 + threads - 1) / threads, threads>>>(W2, dW2, LR, H1 * H2);
        sgd_update_kernel<<<(H2 * H3 + threads - 1) / threads, threads>>>(W3, dW3, LR, H2 * H3);
        sgd_update_kernel<<<(H3 * OUT + threads - 1) / threads, threads>>>(W4, dW4, LR, H3 * OUT);
        
        sgd_update_kernel<<<(H1 + threads - 1) / threads, threads>>>(b1, db1, LR, H1);
        sgd_update_kernel<<<(H2 + threads - 1) / threads, threads>>>(b2, db2, LR, H2);
        sgd_update_kernel<<<(H3 + threads - 1) / threads, threads>>>(b3, db3, LR, H3);
        sgd_update_kernel<<<(OUT + threads - 1) / threads, threads>>>(b4, db4, LR, OUT);

        cudaEventRecord(stop_bw);
        cudaEventSynchronize(stop_bw);

        float ms_fw, ms_bw;
        cudaEventElapsedTime(&ms_fw, start_fw, stop_fw);
        cudaEventElapsedTime(&ms_bw, start_bw, stop_bw);
        total_time_fw += ms_fw;
        total_time_bw += ms_bw;
    }

    std::cout << "Avg Forward Pass Time: " << total_time_fw / ITERATIONS << " ms\n";
    std::cout << "Avg Backward Pass Time: " << total_time_bw / ITERATIONS << " ms\n";

    cudaFree(W1); cudaFree(W2); cudaFree(W3); cudaFree(W4);
    cudaFree(b1); cudaFree(b2); cudaFree(b3); cudaFree(b4);
    cudaFree(X);  cudaFree(y);

    cudaFree(Z1); cudaFree(A1); cudaFree(Z2); cudaFree(A2);
    cudaFree(Z3); cudaFree(A3); cudaFree(Z4);

    cudaFree(dW1); cudaFree(db1); cudaFree(dW2); cudaFree(db2);
    cudaFree(dW3); cudaFree(db3); cudaFree(dW4); cudaFree(db4);

    cudaFree(dZ4); cudaFree(dA3); cudaFree(dZ3); cudaFree(dA2);
    cudaFree(dZ2); cudaFree(dA1); cudaFree(dZ1);

    return 0;
}
