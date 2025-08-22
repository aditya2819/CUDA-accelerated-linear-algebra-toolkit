#ifndef DENSE_MATMUL_CUH
#define DENSE_MATMUL_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <numeric>
#include <algorithm>

#define BLOCK 16
#define WARP_SIZE 32

typedef unsigned int uint;

using namespace std;
using namespace std::chrono;

// CUDA kernel declarations with alpha and beta parameters
__global__ void matMulNaive(const float* A, const float* B, float* C, int N, float alpha, float beta);

__global__ void matMulCoalesced(const float* A, const float* B, float* C, int N, float alpha, float beta);

__global__ void matMulTiledSharedMemory(const float* A, const float* B, float* C, int N, float alpha, float beta);

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void matMul2DTiled(const float* A, const float* B, float* C, int N, float alpha, float beta);

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void matMulVectorized(const float* A, const float* B, float* C, int N, float alpha, float beta);

template <const int WM, const int WN, const int WK, const int WMITER, const int WNITER, const int TM, const int TN>
__global__ void matMulWarpTiled(const float* A, const float* B, float* C, int N, float alpha, float beta);

// Function pointer type for kernel launchers
typedef void (*KernelLauncher)(float*, float*, float*, int, float, float);

// Performance measurement function
double measureKernelPerformance(KernelLauncher setupKernel, 
                               const char* kernelName, 
                               float* d_A, float* d_B, float* d_C, int N, 
                               float alpha, float beta,
                               int numRuns = 10);

// Correctness testing function
bool testCorrectness(KernelLauncher setupKernel,
                    const char* kernelName,
                    float* d_A, float* d_B, float* d_C, int N,
                    const std::vector<float>& cpu_result,
                    float alpha, float beta);

// Kernel launcher functions
void launchNaive(float* d_A, float* d_B, float* d_C, int N, float alpha, float beta);

void launchCoalesced(float* d_A, float* d_B, float* d_C, int N, float alpha, float beta);

void launchTiledShared(float* d_A, float* d_B, float* d_C, int N, float alpha, float beta);

void launch2DTiled(float* d_A, float* d_B, float* d_C, int N, float alpha, float beta);

void launchVectorized(float* d_A, float* d_B, float* d_C, int N, float alpha, float beta);

void launchWarpTiled(float* d_A, float* d_B, float* d_C, int N, float alpha, float beta);

// CPU reference implementation (only computes A*B, no alpha/beta)
void cpuMatMul(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N);

// cuBLAS benchmark function
double benchmarkCuBLAS(float* d_A, float* d_B, float* d_C, int N, float alpha, float beta, int numRuns = 10);

#endif