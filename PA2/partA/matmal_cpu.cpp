#include <iostream>
#include <random>
#include <chrono>
#include <cstdlib>
#include <cmath>

using namespace std;
using namespace std::chrono;

/* ===========================
   WRITE YOUR CPU MATMUL HERE
   =========================== */

void matmul(float* A, float* B, float* C, int N)
{
    // WRITE CPU MATRIX MULTIPLICATION HERE

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

/* =========================== */

/* Reference implementation for verification */
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
        cout << "Usage: ./benchmark_cpu <N>\n";
        return 1;
    }

    int N = atoi(argv[1]);
    size_t bytes = (size_t)N * N * sizeof(float);

    float* A = (float*)malloc(bytes);
    float* B = (float*)malloc(bytes);
    float* C = (float*)malloc(bytes);
    float* C_ref = (float*)malloc(bytes);

    initialize_matrix(A, N);
    initialize_matrix(B, N);

    auto start = high_resolution_clock::now();

    matmul(A, B, C, N);

    auto end = high_resolution_clock::now();

    double time_s = duration<double>(end - start).count();

    /* Verification (outside timing) */
    matmul_reference(A, B, C_ref, N);

    if(verify(C, C_ref, N))
        cout << "Verification PASSED" << endl;
    else
        cout << "Verification FAILED" << endl;

    double checksum = 0.0;
    for(int i = 0; i < N*N; i++)
        checksum += C[i];

    cout << "N = " << N << ", Time(s) = " << time_s << endl;
    cout << C[0] << endl;

    free(A);
    free(B);
    free(C);
    free(C_ref);

    return 0;
}