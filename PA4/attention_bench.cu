// Benchmarks flash.cu's flash_attention_kernel against a standard
// (materialize-the-full-N x N-score-matrix) attention implementation, on
// identical random Q/K/V inputs in a single process, so we get:
//   1. A correctness check (max abs diff between the two outputs) -- flash
//      attention is a numerically-stable *equivalent* of standard softmax
//      attention, so a healthy run should show a tiny diff (float32 rounding
//      only), not something that looks like a bug.
//   2. A clean head-to-head wall-clock + memory comparison, run in the same
//      process/allocator state for both kernels.
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

// ---------------------------------------------------------------------
// Flash attention kernel (same algorithm as flash.cu, after the two fixes:
// the `__syncthreads()` typo and the missing d_m = -inf initialization).
// ---------------------------------------------------------------------
__global__ void init_neg_inf(float* m, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) m[idx] = -1e20f;
}

__global__ void flash_attention_kernel(
    const float* Q, const float* K, const float* V, float* O,
    float* m, float* l,
    int B, int NH, int N, int d, int Tc, int Tr, int Bc, int Br, float scale) {

    int b = blockIdx.x;
    int h = blockIdx.y;
    int tx = threadIdx.x;

    int qkv_base = (b * NH + h) * N * d;
    int lm_base = (b * NH + h) * N;

    extern __shared__ float smem[];
    float* Qi = smem;
    float* Kj = Qi + Br * d;
    float* Vj = Kj + Bc * d;
    float* S = Vj + Bc * d;

    for (int j = 0; j < Tc; j++) {
        for (int x = 0; x < d; x++) {
            int kv_idx = qkv_base + (j * Bc + tx) * d + x;
            if ((j * Bc + tx) < N) {
                Kj[tx * d + x] = K[kv_idx];
                Vj[tx * d + x] = V[kv_idx];
            } else {
                Kj[tx * d + x] = 0.0f;
                Vj[tx * d + x] = 0.0f;
            }
        }
        __syncthreads();

        for (int i = 0; i < Tr; i++) {
            int q_row_global = i * Br + tx;
            bool valid_row = (q_row_global < N);

            if (valid_row) {
                for (int x = 0; x < d; x++) {
                    Qi[tx * d + x] = Q[qkv_base + q_row_global * d + x];
                }
            }

            float m_prev = valid_row ? m[lm_base + q_row_global] : -1e20f;
            float l_prev = valid_row ? l[lm_base + q_row_global] : 0.0f;

            float m_curr = -1e20f;
            if (valid_row) {
                for (int y = 0; y < Bc; y++) {
                    float dot = 0.0f;
                    if ((j * Bc + y) < N) {
                        for (int x = 0; x < d; x++) dot += Qi[tx * d + x] * Kj[y * d + x];
                        dot *= scale;
                    } else {
                        dot = -1e20f;
                    }
                    S[tx * Bc + y] = dot;
                    m_curr = fmaxf(m_curr, dot);
                }
            }

            float l_curr = 0.0f;
            if (valid_row) {
                for (int y = 0; y < Bc; y++) {
                    float val = expf(S[tx * Bc + y] - m_curr);
                    if ((j * Bc + y) >= N) val = 0.0f;
                    S[tx * Bc + y] = val;
                    l_curr += val;
                }
            }

            if (valid_row) {
                float m_new = fmaxf(m_prev, m_curr);
                float exp_prev = expf(m_prev - m_new);
                float exp_curr = expf(m_curr - m_new);
                float l_new = l_prev * exp_prev + l_curr * exp_curr;

                for (int x = 0; x < d; x++) {
                    float pv = 0.0f;
                    for (int y = 0; y < Bc; y++) pv += S[tx * Bc + y] * Vj[y * d + x];
                    float o_prev = O[qkv_base + q_row_global * d + x];
                    float o_new = (l_prev * exp_prev * o_prev + exp_curr * pv) / l_new;
                    O[qkv_base + q_row_global * d + x] = o_new;
                }
                m[lm_base + q_row_global] = m_new;
                l[lm_base + q_row_global] = l_new;
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------
// Standard attention: full N x N score matrix, materialized in global mem.
// ---------------------------------------------------------------------
// blockIdx.z indexes (batch*head) so this scales the same way flash's
// dim3 grid(B, NH) does -- the point of this experiment is to give both
// kernels the same batch*head-parallel grid and see what happens.
__global__ void compute_scores(const float* Q, const float* K, float* S,
                                int N, int d, float scale) {
    int bh = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N) {
        const float* Qb = Q + (size_t)bh * N * d;
        const float* Kb = K + (size_t)bh * N * d;
        float dot = 0.0f;
        for (int x = 0; x < d; x++) dot += Qb[i * d + x] * Kb[j * d + x];
        S[(size_t)bh * N * N + (size_t)i * N + j] = dot * scale;
    }
}

__global__ void softmax_rows(float* S, int N) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int bh = blockIdx.y;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    float* row_ptr = S + (size_t)bh * N * N + (size_t)row * N;

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

__global__ void compute_output(const float* S, const float* V, float* O,
                                int N, int d) {
    int bh = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && x < d) {
        const float* Sb = S + (size_t)bh * N * N;
        const float* Vb = V + (size_t)bh * N * d;
        float acc = 0.0f;
        for (int j = 0; j < N; j++) acc += Sb[i * N + j] * Vb[j * d + x];
        O[(size_t)bh * N * d + i * d + x] = acc;
    }
}

__global__ void max_abs_diff_kernel(const float* A, const float* B, float* out, int size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float v = 0.0f;
    if (idx < size) v = fabsf(A[idx] - B[idx]);
    sdata[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) atomicMax((int*)out, __float_as_int(sdata[0]));
}

int main(int argc, char *argv[]) {
    int N = 1024;
    if (argc >= 2) N = atoi(argv[1]);
    unsigned seed = 42;
    if (argc >= 3) seed = (unsigned)atoi(argv[2]);
    int B = 1, NH = 1;
    if (argc >= 4) B = atoi(argv[3]);
    if (argc >= 5) NH = atoi(argv[4]);

    int d = 64;
    int Br = 32, Bc = 32;
    int Tr = (N + Br - 1) / Br;
    int Tc = (N + Bc - 1) / Bc;
    float scale = 1.0f / sqrtf((float)d);

    size_t qkv_size = (size_t)B * NH * N * d * sizeof(float);
    size_t lm_size = (size_t)B * NH * N * sizeof(float);
    size_t s_size = (size_t)B * NH * N * N * sizeof(float);

    float *h_Q, *h_K, *h_V;
    cudaMallocHost(&h_Q, qkv_size);
    cudaMallocHost(&h_K, qkv_size);
    cudaMallocHost(&h_V, qkv_size);
    srand(seed);
    for (size_t i = 0; i < (size_t)B * NH * N * d; ++i) {
        h_Q[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
        h_K[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
        h_V[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
    }

    float *d_Q, *d_K, *d_V;
    cudaMalloc(&d_Q, qkv_size);
    cudaMalloc(&d_K, qkv_size);
    cudaMalloc(&d_V, qkv_size);
    cudaMemcpy(d_Q, h_Q, qkv_size, cudaMemcpyDefault);
    cudaMemcpy(d_K, h_K, qkv_size, cudaMemcpyDefault);
    cudaMemcpy(d_V, h_V, qkv_size, cudaMemcpyDefault);

    // ---- Flash attention ----
    float *d_O_flash, *d_m, *d_l;
    cudaMalloc(&d_O_flash, qkv_size);
    cudaMalloc(&d_m, lm_size);
    cudaMalloc(&d_l, lm_size);
    cudaMemset(d_O_flash, 0, qkv_size);
    cudaMemset(d_l, 0, lm_size);
    init_neg_inf<<<(B * NH * N + 255) / 256, 256>>>(d_m, B * NH * N);

    dim3 grid_flash(B, NH);
    dim3 block_flash(Br);
    size_t smem = (Br * d + 2 * Bc * d + Br * Bc) * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();
    cudaEventRecord(start);
    flash_attention_kernel<<<grid_flash, block_flash, smem>>>(
        d_Q, d_K, d_V, d_O_flash, d_m, d_l, B, NH, N, d, Tc, Tr, Bc, Br, scale);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float flash_ms = 0;
    cudaEventElapsedTime(&flash_ms, start, stop);
    cudaError_t flash_err = cudaGetLastError();

    // ---- Standard attention ----
    float *d_O_std, *d_S;
    cudaMalloc(&d_O_std, qkv_size);
    cudaError_t s_alloc_err = cudaMalloc(&d_S, s_size);
    if (s_alloc_err != cudaSuccess) {
        cout << "N=" << N << " B=" << B << " NH=" << NH
             << " standard_score_matrix_bytes=" << s_size
             << " standard_alloc_error=" << cudaGetErrorString(s_alloc_err)
             << endl;
        return 1;
    }

    dim3 block2d(16, 16);
    dim3 grid_scores((N + 15) / 16, (N + 15) / 16, B * NH);
    dim3 grid_out((d + 15) / 16, (N + 15) / 16, B * NH);
    dim3 grid_softmax(N, B * NH);
    int softmax_threads = 256;

    cudaDeviceSynchronize();
    cudaEventRecord(start);
    compute_scores<<<grid_scores, block2d>>>(d_Q, d_K, d_S, N, d, scale);
    softmax_rows<<<grid_softmax, softmax_threads, softmax_threads * sizeof(float)>>>(d_S, N);
    compute_output<<<grid_out, block2d>>>(d_S, d_V, d_O_std, N, d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float std_ms = 0;
    cudaEventElapsedTime(&std_ms, start, stop);
    cudaError_t std_err = cudaGetLastError();

    // ---- Correctness: max abs diff between the two outputs ----
    float *d_diff;
    cudaMalloc(&d_diff, sizeof(float));
    float zero = 0.0f;
    cudaMemcpy(d_diff, &zero, sizeof(float), cudaMemcpyDefault);
    int total = B * NH * N * d;
    max_abs_diff_kernel<<<(total + 255) / 256, 256, 256 * sizeof(float)>>>(d_O_flash, d_O_std, d_diff, total);
    float max_diff = 0.0f;
    cudaMemcpy(&max_diff, d_diff, sizeof(float), cudaMemcpyDefault);

    size_t free_b, total_b;
    cudaMemGetInfo(&free_b, &total_b);

    cout << "N=" << N
         << " B=" << B << " NH=" << NH
         << " flash_ms=" << (flash_ms)
         << " standard_ms=" << (std_ms)
         << " speedup=" << (std_ms / flash_ms)
         << " max_abs_diff=" << max_diff
         << " flash_err=" << cudaGetErrorString(flash_err)
         << " std_err=" << cudaGetErrorString(std_err)
         << " flash_aux_bytes=" << (2 * lm_size)
         << " standard_score_matrix_bytes=" << s_size
         << " gpu_free_bytes=" << free_b
         << " gpu_total_bytes=" << total_b
         << endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O_flash); cudaFree(d_m); cudaFree(d_l);
    cudaFree(d_O_std); cudaFree(d_S); cudaFree(d_diff);
    cudaFreeHost(h_Q); cudaFreeHost(h_K); cudaFreeHost(h_V);

    return 0;
}
