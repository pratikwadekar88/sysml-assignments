#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <cstdlib>

void init_matrix(float* mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX * 0.01f;
    }
}

__global__ void matmul(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmul_transA(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[k * M + row] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmul_transB(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[col * K + k];
        }
        C[row * N + col] = sum;
    }
}

__global__ void add_bias(float* Z, float* b, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        Z[row * N + col] += b[col];
    }
}

__global__ void relu(float* A, float* Z, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] = Z[idx] > 0.0f ? Z[idx] : 0.0f;
    }
}

__global__ void relu_backward(float* dZ, float* dA, float* Z, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dZ[idx] = dA[idx] * (Z[idx] > 0.0f ? 1.0f : 0.0f);
    }
}

__global__ void mse_backward(float* dZ4, float* Z4, float* Y, int N, int OUT) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = N * OUT;
    if (idx < size) {
        dZ4[idx] = 2.0f * (Z4[idx] - Y[idx]) / size;
    }
}

__global__ void bias_backward(float* db, float* dZ, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        float sum = 0.0f;
        for (int row = 0; row < M; ++row) {
            sum += dZ[row * N + col];
        }
        db[col] = sum;
    }
}

__global__ void sgd_update(float* param, float* grad, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        param[idx] -= lr * grad[idx];
    }
}

void forward_pass(float* X, float* W1, float* b1, float* W2, float* b2, 
                  float* W3, float* b3, float* W4, float* b4, 
                  float* A2_checkpoint, float* Z4_out, 
                  int N, int IN, int H1, int H2, int H3, int OUT) {
    float *Z1, *A1, *Z2, *Z3, *A3;
    cudaMalloc(&Z1, N * H1 * sizeof(float));
    cudaMalloc(&A1, N * H1 * sizeof(float));
    cudaMalloc(&Z2, N * H2 * sizeof(float));
    cudaMalloc(&Z3, N * H3 * sizeof(float));
    cudaMalloc(&A3, N * H3 * sizeof(float));

    dim3 block(16, 16);
    dim3 grid_H1((H1 + 15) / 16, (N + 15) / 16);
    dim3 grid_H2((H2 + 15) / 16, (N + 15) / 16);
    dim3 grid_H3((H3 + 15) / 16, (N + 15) / 16);
    dim3 grid_OUT((OUT + 15) / 16, (N + 15) / 16);

    matmul<<<grid_H1, block>>>(X, W1, Z1, N, H1, IN);
    add_bias<<<grid_H1, block>>>(Z1, b1, N, H1);
    relu<<<(N * H1 + 255) / 256, 256>>>(A1, Z1, N * H1);

    matmul<<<grid_H2, block>>>(A1, W2, Z2, N, H2, H1);
    add_bias<<<grid_H2, block>>>(Z2, b2, N, H2);
    relu<<<(N * H2 + 255) / 256, 256>>>(A2_checkpoint, Z2, N * H2);

    matmul<<<grid_H3, block>>>(A2_checkpoint, W3, Z3, N, H3, H2);
    add_bias<<<grid_H3, block>>>(Z3, b3, N, H3);
    relu<<<(N * H3 + 255) / 256, 256>>>(A3, Z3, N * H3);

    matmul<<<grid_OUT, block>>>(A3, W4, Z4_out, N, OUT, H3);
    add_bias<<<grid_OUT, block>>>(Z4_out, b4, N, OUT);

    cudaFree(Z1);
    cudaFree(A1);
    cudaFree(Z2);
    cudaFree(Z3);
    cudaFree(A3);
}

void backward_pass_checkpointing(float* X, float* Y, 
                                 float* W1, float* b1, float* dW1, float* db1,
                                 float* W2, float* b2, float* dW2, float* db2,
                                 float* W3, float* b3, float* dW3, float* db3,
                                 float* W4, float* b4, float* dW4, float* db4,
                                 float* A2_ckpt, float* Z4_out,
                                 int N, int IN, int H1, int H2, int H3, int OUT, float lr) {
    float *Z1, *A1, *Z2, *Z3, *A3;
    cudaMalloc(&Z1, N * H1 * sizeof(float));
    cudaMalloc(&A1, N * H1 * sizeof(float));
    cudaMalloc(&Z2, N * H2 * sizeof(float));
    cudaMalloc(&Z3, N * H3 * sizeof(float));
    cudaMalloc(&A3, N * H3 * sizeof(float));

    dim3 block(16, 16);
    dim3 grid_H1((H1 + 15) / 16, (N + 15) / 16);
    dim3 grid_H2((H2 + 15) / 16, (N + 15) / 16);
    dim3 grid_H3((H3 + 15) / 16, (N + 15) / 16);

    matmul<<<grid_H1, block>>>(X, W1, Z1, N, H1, IN);
    add_bias<<<grid_H1, block>>>(Z1, b1, N, H1);
    relu<<<(N * H1 + 255) / 256, 256>>>(A1, Z1, N * H1);

    matmul<<<grid_H2, block>>>(A1, W2, Z2, N, H2, H1);
    add_bias<<<grid_H2, block>>>(Z2, b2, N, H2);

    matmul<<<grid_H3, block>>>(A2_ckpt, W3, Z3, N, H3, H2);
    add_bias<<<grid_H3, block>>>(Z3, b3, N, H3);
    relu<<<(N * H3 + 255) / 256, 256>>>(A3, Z3, N * H3);

    float *dZ4, *dA3, *dZ3, *dA2, *dZ2, *dA1, *dZ1;
    cudaMalloc(&dZ4, N * OUT * sizeof(float));
    cudaMalloc(&dA3, N * H3 * sizeof(float));
    cudaMalloc(&dZ3, N * H3 * sizeof(float));
    cudaMalloc(&dA2, N * H2 * sizeof(float));
    cudaMalloc(&dZ2, N * H2 * sizeof(float));
    cudaMalloc(&dA1, N * H1 * sizeof(float));
    cudaMalloc(&dZ1, N * H1 * sizeof(float));

    mse_backward<<<(N * OUT + 255) / 256, 256>>>(dZ4, Z4_out, Y, N, OUT);

    dim3 grid_dW4((OUT + 15) / 16, (H3 + 15) / 16);
    dim3 grid_dA3((H3 + 15) / 16, (N + 15) / 16);
    matmul_transA<<<grid_dW4, block>>>(A3, dZ4, dW4, H3, OUT, N);
    bias_backward<<<(OUT + 255) / 256, 256>>>(db4, dZ4, N, OUT);
    matmul_transB<<<grid_dA3, block>>>(dZ4, W4, dA3, N, H3, OUT);

    relu_backward<<<(N * H3 + 255) / 256, 256>>>(dZ3, dA3, Z3, N * H3);
    dim3 grid_dW3((H3 + 15) / 16, (H2 + 15) / 16);
    dim3 grid_dA2((H2 + 15) / 16, (N + 15) / 16);
    matmul_transA<<<grid_dW3, block>>>(A2_ckpt, dZ3, dW3, H2, H3, N);
    bias_backward<<<(H3 + 255) / 256, 256>>>(db3, dZ3, N, H3);
    matmul_transB<<<grid_dA2, block>>>(dZ3, W3, dA2, N, H2, H3);

    relu_backward<<<(N * H2 + 255) / 256, 256>>>(dZ2, dA2, Z2, N * H2);
    dim3 grid_dW2((H2 + 15) / 16, (H1 + 15) / 16);
    dim3 grid_dA1((H1 + 15) / 16, (N + 15) / 16);
    matmul_transA<<<grid_dW2, block>>>(A1, dZ2, dW2, H1, H2, N);
    bias_backward<<<(H2 + 255) / 256, 256>>>(db2, dZ2, N, H2);
    matmul_transB<<<grid_dA1, block>>>(dZ2, W2, dA1, N, H1, H2);

    relu_backward<<<(N * H1 + 255) / 256, 256>>>(dZ1, dA1, Z1, N * H1);
    dim3 grid_dW1((H1 + 15) / 16, (IN + 15) / 16);
    matmul_transA<<<grid_dW1, block>>>(X, dZ1, dW1, IN, H1, N);
    bias_backward<<<(H1 + 255) / 256, 256>>>(db1, dZ1, N, H1);

    cudaFree(Z1); cudaFree(A1); cudaFree(Z2); cudaFree(Z3); cudaFree(A3);
    cudaFree(dZ4); cudaFree(dA3); cudaFree(dZ3); cudaFree(dA2); cudaFree(dZ2); cudaFree(dA1); cudaFree(dZ1);

    sgd_update<<<(IN * H1 + 255) / 256, 256>>>(W1, dW1, lr, IN * H1);
    sgd_update<<<(H1 + 255) / 256, 256>>>(b1, db1, lr, H1);
    sgd_update<<<(H1 * H2 + 255) / 256, 256>>>(W2, dW2, lr, H1 * H2);
    sgd_update<<<(H2 + 255) / 256, 256>>>(b2, db2, lr, H2);
    sgd_update<<<(H2 * H3 + 255) / 256, 256>>>(W3, dW3, lr, H2 * H3);
    sgd_update<<<(H3 + 255) / 256, 256>>>(b3, db3, lr, H3);
    sgd_update<<<(H3 * OUT + 255) / 256, 256>>>(W4, dW4, lr, H3 * OUT);
    sgd_update<<<(OUT + 255) / 256, 256>>>(b4, db4, lr, OUT);
}

int main() {
    int N = 1024, IN = 512, H1 = 2048, H2 = 2048, H3 = 2048, OUT = 512;
    float lr = 0.001f;

    float *X, *Y;
    float *W1, *b1, *dW1, *db1;
    float *W2, *b2, *dW2, *db2;
    float *W3, *b3, *dW3, *db3;
    float *W4, *b4, *dW4, *db4;
    float *A2_ckpt, *Z4_out;

    cudaMalloc(&X, N * IN * sizeof(float));
    cudaMalloc(&Y, N * OUT * sizeof(float));

    cudaMalloc(&W1, IN * H1 * sizeof(float)); cudaMalloc(&b1, H1 * sizeof(float));
    cudaMalloc(&dW1, IN * H1 * sizeof(float)); cudaMalloc(&db1, H1 * sizeof(float));

    cudaMalloc(&W2, H1 * H2 * sizeof(float)); cudaMalloc(&b2, H2 * sizeof(float));
    cudaMalloc(&dW2, H1 * H2 * sizeof(float)); cudaMalloc(&db2, H2 * sizeof(float));

    cudaMalloc(&W3, H2 * H3 * sizeof(float)); cudaMalloc(&b3, H3 * sizeof(float));
    cudaMalloc(&dW3, H2 * H3 * sizeof(float)); cudaMalloc(&db3, H3 * sizeof(float));

    cudaMalloc(&W4, H3 * OUT * sizeof(float)); cudaMalloc(&b4, OUT * sizeof(float));
    cudaMalloc(&dW4, H3 * OUT * sizeof(float)); cudaMalloc(&db4, OUT * sizeof(float));

    cudaMalloc(&A2_ckpt, N * H2 * sizeof(float));
    cudaMalloc(&Z4_out, N * OUT * sizeof(float));

    float *h_X = new float[N * IN];
    float *h_Y = new float[N * OUT];
    float *h_W1 = new float[IN * H1];
    float *h_b1 = new float[H1];
    float *h_W2 = new float[H1 * H2];
    float *h_b2 = new float[H2];
    float *h_W3 = new float[H2 * H3];
    float *h_b3 = new float[H3];
    float *h_W4 = new float[H3 * OUT];
    float *h_b4 = new float[OUT];

    init_matrix(h_X, N * IN);
    init_matrix(h_Y, N * OUT);
    init_matrix(h_W1, IN * H1);
    init_matrix(h_b1, H1);
    init_matrix(h_W2, H1 * H2);
    init_matrix(h_b2, H2);
    init_matrix(h_W3, H2 * H3);
    init_matrix(h_b3, H3);
    init_matrix(h_W4, H3 * OUT);
    init_matrix(h_b4, OUT);

    cudaMemcpy(X, h_X, N * IN * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Y, h_Y, N * OUT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(W1, h_W1, IN * H1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b1, h_b1, H1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(W2, h_W2, H1 * H2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b2, h_b2, H2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(W3, h_W3, H2 * H3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b3, h_b3, H3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(W4, h_W4, H3 * OUT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b4, h_b4, OUT * sizeof(float), cudaMemcpyHostToDevice);

    delete[] h_X; delete[] h_Y; 
    delete[] h_W1; delete[] h_b1; 
    delete[] h_W2; delete[] h_b2; 
    delete[] h_W3; delete[] h_b3; 
    delete[] h_W4; delete[] h_b4;

    size_t free_byte;
    size_t total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    std::cout << "Free Memory: " << free_byte << " bytes\n";
    std::cout << "Total Memory: " << total_byte << " bytes\n";

    forward_pass(X, W1, b1, W2, b2, W3, b3, W4, b4, A2_ckpt, Z4_out, N, IN, H1, H2, H3, OUT);
    cudaDeviceSynchronize();

    cudaEvent_t start_fwd, stop_fwd, start_bwd, stop_bwd;
    cudaEventCreate(&start_fwd); cudaEventCreate(&stop_fwd);
    cudaEventCreate(&start_bwd); cudaEventCreate(&stop_bwd);

    int iterations = 100;
    float total_fwd_time = 0.0f;
    float total_bwd_time = 0.0f;

    for (int i = 0; i < iterations; ++i) {
        cudaEventRecord(start_fwd);
        forward_pass(X, W1, b1, W2, b2, W3, b3, W4, b4, A2_ckpt, Z4_out, N, IN, H1, H2, H3, OUT);
        cudaEventRecord(stop_fwd);
        cudaEventSynchronize(stop_fwd);

        float fwd_ms;
        cudaEventElapsedTime(&fwd_ms, start_fwd, stop_fwd);
        total_fwd_time += fwd_ms;

        cudaEventRecord(start_bwd);
        backward_pass_checkpointing(X, Y, W1, b1, dW1, db1, W2, b2, dW2, db2, W3, b3, dW3, db3, W4, b4, dW4, db4, A2_ckpt, Z4_out, N, IN, H1, H2, H3, OUT, lr);
        cudaEventRecord(stop_bwd);
        cudaEventSynchronize(stop_bwd);

        float bwd_ms;
        cudaEventElapsedTime(&bwd_ms, start_bwd, stop_bwd);
        total_bwd_time += bwd_ms;
    }

    std::cout << "Avg Forward Time: " << total_fwd_time / iterations << " ms\n";
    std::cout << "Avg Backward Time: " << total_bwd_time / iterations << " ms\n";

    cudaFree(X); cudaFree(Y);
    cudaFree(W1); cudaFree(b1); cudaFree(dW1); cudaFree(db1);
    cudaFree(W2); cudaFree(b2); cudaFree(dW2); cudaFree(db2);
    cudaFree(W3); cudaFree(b3); cudaFree(dW3); cudaFree(db3);
    cudaFree(W4); cudaFree(b4); cudaFree(dW4); cudaFree(db4);
    cudaFree(A2_ckpt); cudaFree(Z4_out);

    return 0;
}
