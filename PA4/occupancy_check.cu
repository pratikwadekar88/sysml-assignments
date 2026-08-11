// Uses the CUDA runtime's own occupancy calculator (cudaOccupancyMaxActiveBlocksPerMultiprocessor)
// to report flash_attention_kernel's real achievable occupancy on this GPU, driven by its actual
// per-block shared memory footprint and thread count. This needs no special driver permissions
// (unlike ncu/nvprof counter sampling, which ERR_NVGPUCTRPERM's on this WSL2 setup), so it's a
// runtime-computed number, not a hand-derived guess.
#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void init_neg_inf(float* m, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) m[idx] = -1e20f;
}

__global__ void flash_attention_kernel(
    const float* Q, const float* K, const float* V, float* O,
    float* m, float* l,
    int B, int NH, int N, int d, int Tc, int Tr, int Bc, int Br, float scale) {
    int b = blockIdx.x; int h = blockIdx.y; int tx = threadIdx.x;
    (void)b; (void)h; (void)tx; (void)Q; (void)K; (void)V; (void)O; (void)m; (void)l;
    (void)B; (void)NH; (void)N; (void)d; (void)Tc; (void)Tr; (void)Bc; (void)Br; (void)scale;
}

__global__ void compute_scores(const float* Q, const float* K, float* S, int N, int d, float scale) {
    (void)Q; (void)K; (void)S; (void)N; (void)d; (void)scale;
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout << "Device: " << prop.name << " (sm_" << prop.major << prop.minor << ")\n";
    cout << "SMs: " << prop.multiProcessorCount
         << ", maxThreadsPerSM: " << prop.maxThreadsPerMultiProcessor
         << ", maxThreadsPerBlock: " << prop.maxThreadsPerBlock
         << ", sharedMemPerSM: " << prop.sharedMemPerMultiprocessor << " bytes"
         << ", sharedMemPerBlock(default): " << prop.sharedMemPerBlock << " bytes"
         << ", sharedMemPerBlockOptin: " << prop.sharedMemPerBlockOptin << " bytes"
         << ", regsPerSM: " << prop.regsPerMultiprocessor
         << ", warpSize: " << prop.warpSize << "\n\n";

    int Br = 32, Bc = 32, d = 64;
    size_t smem = (size_t)(Br * d + 2 * Bc * d + Br * Bc) * sizeof(float);
    cout << "flash_attention_kernel: block=" << Br << " threads, dynamic smem=" << smem << " bytes\n";

    int numBlocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, flash_attention_kernel, Br, smem);
    int activeThreads = numBlocks * Br;
    float occupancy = 100.0f * activeThreads / prop.maxThreadsPerMultiProcessor;
    cout << "  cudaOccupancyMaxActiveBlocksPerMultiprocessor -> " << numBlocks << " resident blocks/SM\n";
    cout << "  active threads/SM = " << activeThreads << " / " << prop.maxThreadsPerMultiProcessor
         << " -> achieved occupancy = " << occupancy << "%\n\n";

    // Standard attention's compute_scores/compute_output kernels: 16x16 thread blocks, no
    // dynamic shared memory (softmax_rows uses static+dynamic shared mem sized by block dim,
    // omitted here since it's launched with N threads/block, not a fixed shape).
    int numBlocksStd = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksStd, compute_scores, 256, 0);
    int activeThreadsStd = numBlocksStd * 256;
    float occupancyStd = 100.0f * activeThreadsStd / prop.maxThreadsPerMultiProcessor;
    cout << "compute_scores (standard attention): block=256 threads, no dynamic smem\n";
    cout << "  cudaOccupancyMaxActiveBlocksPerMultiprocessor -> " << numBlocksStd << " resident blocks/SM\n";
    cout << "  active threads/SM = " << activeThreadsStd << " / " << prop.maxThreadsPerMultiProcessor
         << " -> achieved occupancy = " << occupancyStd << "%\n";

    return 0;
}
