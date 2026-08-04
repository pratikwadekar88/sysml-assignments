#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

__global__ void flash_attention_kernel(
    const float* Q, const float* K, const float* V, float* O, 
    float* m, float* l, 
    int B, int NH, int N, int d, int Tc, int Tr, int Bc, int Br, float scale) {
    
    // Block and thread indices
    int b = blockIdx.x; 
    int h = blockIdx.y; 
    int tx = threadIdx.x; // tx in {0, 1, ..., Br-1}

    // Global memory base offsets
    int qkv_base = (b * NH + h) * N * d; 
    int lm_base = (b * NH + h) * N; 

    // Shared Memory Layout
    extern __shared__ float smem[];
    float* Qi = smem; 
    float* Kj = Qi + Br * d; 
    float* Vj = Kj + Bc * d; 
    float* S = Vj + Bc * d; 

    // Outer Loop: Iterating Over K/V Tiles (j=0...Tc-1)
    for (int j = 0; j < Tc; j++) {
        
        // Load K and V tiles into shared memory cooperatively
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
        
        __syncthreads(); // BARRIER 1: all Bc rows of K/V are in smem

        // Inner Loop: Iterating Over Q Tiles (i=0...Tr-1)
        for (int i = 0; i < Tr; i++) {
            int q_row_global = i * Br + tx;
            bool valid_row = (q_row_global < N);

            // Step 1: Load Q tile
            // Thread tx loads its own row from global memory into shared memory.
            if (valid_row) {
                for (int x = 0; x < d; x++) {
                    Qi[tx * d + x] = Q[qkv_base + q_row_global * d + x];
                }
            }

            // Step 2: Read running statistics
            // Initialize m to -infinity and l to 0 if it's the first K/V tile, otherwise read from global memory.
            float m_prev = valid_row ? m[lm_base + q_row_global] : -1e20f;
            float l_prev = valid_row ? l[lm_base + q_row_global] : 0.0f;

            // Step 3: Compute dot products and tile maximum
            float m_curr = -1e20f;
            if (valid_row) {
                for (int y = 0; y < Bc; y++) {
                    float dot = 0.0f;
                    if ((j * Bc + y) < N) {
                        for (int x = 0; x < d; x++) {
                            dot += Qi[tx * d + x] * Kj[y * d + x];
                        }
                        dot *= scale; // Scale by 1/sqrt(d)
                    } else {
                        dot = -1e20f; 
                    }
                    S[tx * Bc + y] = dot;
                    m_curr = fmaxf(m_curr, dot);
                }
            }

            // Step 4: Exponentiate and accumulate tile sum
            float l_curr = 0.0f;
            if (valid_row) {
                for (int y = 0; y < Bc; y++) {
                    float val = expf(S[tx * Bc + y] - m_curr);
                    if ((j * Bc + y) >= N) val = 0.0f; 
                    S[tx * Bc + y] = val; // S now holds P_ij
                    l_curr += val;
                }
            }

            // Step 5: Online softmax update and output rescaling
            if (valid_row) {
                float m_new = fmaxf(m_prev, m_curr);
                float exp_prev = expf(m_prev - m_new);
                float exp_curr = expf(m_curr - m_new);
                float l_new = l_prev * exp_prev + l_curr * exp_curr;

                for (int x = 0; x < d; x++) {
                    // Compute new PV contribution for this element
                    float pv = 0.0f;
                    for (int y = 0; y < Bc; y++) {
                        pv += S[tx * Bc + y] * Vj[y * d + x];
                    }
                    
                    // Rescale accumulated output and add new contribution
                    float o_prev = O[qkv_base + q_row_global * d + x];
                    float o_new = (l_prev * exp_prev * o_prev + exp_curr * pv) / l_new;
                    
                    // Write back to global memory
                    O[qkv_base + q_row_global * d + x] = o_new;
                }

                // Update running statistics in global memory
                m[lm_base + q_row_global] = m_new;
                l[lm_base + q_row_global] = l_new;
            }
        }
        
        __syncthreads(  ; // BARRIER 2: prevent overwriting K/V in shared memory before all inner loops finish
    }
}

// __global__ void flash_attention_kernel(
//     const float* Q, const float* K, const float* V, float* O, 
//     float* m, float* l, 
//     int B, int NH, int N, int d, int Tc, int Tr, int Bc, int Br, float scale) {
    
//     // Block and thread indices
//     int b = blockIdx.x
//     int h = blockIdx.y
//     int tx = threadIdx.x

//     // Global memory base offsets
//     int qkv_base = (b*NH+h)*N*d;
//     int lm_base = (b*NH+h) * N;

//     // Shared Memory Layout
//     __shared__ float * smem[];
//     float* Qi = smem;
//     float* Kj = Qi + Br*d;
//     float* Vj = Kj + Bc*d;
//     float* S  = Vj + Bc*d;

//     // Outer Loop: Iterating Over K/V Tiles (j=0...Tc-1)
//     for(int j =0;j<Tc;j++)

//         for(int x = 0;x<d;x++){
//             int kv_idx = qkv_base + (j*Bc+tx)*d + x;
//             Kj[tx*d+x] = K[kv_idx];
//             Vj[tx*d+x] = V[kv_idx];
//         }    

//         __syncthreads(); // BARRIER 1: all Bc rows of K/V are in smem

//         // Inner Loop: Iterating Over Q Tiles (i=0...Tr-1)
//         for(int i=0;i<Tr;i++)
//             int q_row_global = (i*Br+tx)
//             bool valid_row = (q_row_global<N)
//             // Step 1: Load Q tile
//             // Thread tx loads its own row from global memory into shared memory.
//             if(valid_row){
//                 for(int x = 0;x<d;x++){
//                     Qi[tx*d+x] = Q[q_row_global*d+x];
//                 } 
//             }

//             // Step 2: Read running statistics
//             // Initialize m to -infinity and l to 0 if it's the first K/V tile, otherwise read from global memory.
            
//             // Step 3: Compute dot products and tile maximum
            

//             // Step 4: Exponentiate and accumulate tile sum
            

//             // Step 5: Online softmax update and output rescaling
            

                
//                     // Compute new PV contribution for this element
                    
                    
//                     // Rescale accumulated output and add new contribution
                   
                    
//                     // Write back to global memory
                    
//                 }

//                 // Update running statistics in global memory
//                `
//             }
//         }
        
//         __syncthreads(  ; // BARRIER 2: prevent overwriting K/V in shared memory before all inner loops finish
//     }
// }

int main(int argc, char *argv[]) {
    int N = 1024; // Default sequence length
    if (argc >= 2) {
        N = atoi(argv[1]);
    }

    // FlashAttention specific dimensions 
    int B = 1;      // Batch size
    int NH = 1;     // Number of heads
    int d = 64;     // Head dimension
    int Br = 32;    // Rows per Q tile [cite: 19]
    int Bc = 32;    // Rows per K/V tile [cite: 19]
    int Tr = (N + Br - 1) / Br; // [cite: 20]
    int Tc = (N + Bc - 1) / Bc; // [cite: 23]
    float scale = 1.0f / sqrt(d);

    size_t qkv_size = B * NH * N * d * sizeof(float);
    size_t lm_size = B * NH * N * sizeof(float);

    // Allocate memory on the host
    float *h_Q, *h_K, *h_V, *h_O;
    cudaMallocHost(&h_Q, qkv_size);
    cudaMallocHost(&h_K, qkv_size);
    cudaMallocHost(&h_V, qkv_size);
    cudaMallocHost(&h_O, qkv_size);

    // Initialize host matrices
    for (int i = 0; i < B * NH * N * d; ++i) {
        h_Q[i] = 0.1f; // Sample values
        h_K[i] = 0.1f;
        h_V[i] = 0.1f;
    }

    // Allocate memory on the device
    float *d_Q, *d_K, *d_V, *d_O, *d_m, *d_l;
    cudaMalloc(&d_Q, qkv_size);
    cudaMalloc(&d_K, qkv_size);
    cudaMalloc(&d_V, qkv_size);
    cudaMalloc(&d_O, qkv_size);
    cudaMalloc(&d_m, lm_size);
    cudaMalloc(&d_l, lm_size);

    // Copy matrices from host to device
    cudaMemcpy(d_Q, h_Q, qkv_size, cudaMemcpyDefault);
    cudaMemcpy(d_K, h_K, qkv_size, cudaMemcpyDefault);
    cudaMemcpy(d_V, h_V, qkv_size, cudaMemcpyDefault);
    
    // Initialize O and l on device [cite: 90]
    cudaMemset(d_O, 0, qkv_size);
    cudaMemset(d_l, 0, lm_size);

    // Define grid and block dimensions 
    dim3 grid(B, NH); // total of B*NH thread blocks [cite: 32, 33]
    dim3 block(Br);   // 32 threads per block [cite: 32, 33]
    
    // Total shared memory size passed at launch [cite: 75]
    size_t smem = (Br * d + 2 * Bc * d + Br * Bc) * sizeof(float); // [cite: 76]

    // --- TIMING START ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    flash_attention_kernel<<<grid, block, smem>>>(d_Q, d_K, d_V, d_O, d_m, d_l, B, NH, N, d, Tc, Tr, Bc, Br, scale);
    cudaEventRecord(stop);

    // Wait for the stop event to complete
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // --- TIMING END ---

    // Copy the result back to host
    cudaMemcpy(h_O, d_O, qkv_size, cudaMemcpyDefault);

    // Output results
    cout << "FlashAttention Kernel Time: " << milliseconds / 1000.0f << " s" << endl;

    // Free device memory
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_m);
    cudaFree(d_l);

    // Free host memory
    cudaFreeHost(h_Q);
    cudaFreeHost(h_K);
    cudaFreeHost(h_V);
    cudaFreeHost(h_O);

    return 0;
}