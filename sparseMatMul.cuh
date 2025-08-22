// sparseMatMul.cuh - Header file for CUDA CSR Sparse Matrix-Vector Multiplication
#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cusparse.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <string>
#include <random>
#include <set>
#include <algorithm>
#include <functional>
#include <numeric>
#include <cmath>

using namespace std;
using namespace std::chrono;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Forward declarations for all kernel launch functions
void launchBasicSpMV(const int* d_row_ptr, const int* d_col_idx, const float* d_values,
                     const float* d_x, float* d_y, int num_rows);

void launchWarpReductionSpMV(const int* d_row_ptr, const int* d_col_idx, const float* d_values,
                            const float* d_x, float* d_y, int num_rows);

void launchCacheOptimizedSpMV(const int* d_row_ptr, const int* d_col_idx, const float* d_values,
                             const float* d_x, float* d_y, int num_rows);

void launchVectorizedSpMV(const int* d_row_ptr, const int* d_col_idx, const float* d_values,
                         const float* d_x, float* d_y, int num_rows);

void launchSharedMemorySpMV(const int* d_row_ptr, const int* d_col_idx, const float* d_values,
                           const float* d_x, float* d_y, int num_rows);

void launchPrefetchSpMV(const int* d_row_ptr, const int* d_col_idx, const float* d_values,
                       const float* d_x, float* d_y, int num_rows);

void launchAdaptiveSpMV(const int* d_row_ptr, const int* d_col_idx, const float* d_values,
                       const float* d_x, float* d_y, int num_rows);

void launchUltimateSpMV(const int* d_row_ptr, const int* d_col_idx, const float* d_values,
                       const float* d_x, float* d_y, int num_rows);

// Utility functions
void generateCSRMatrix(std::vector<int>& row_ptr, std::vector<int>& col_idx, 
                      std::vector<float>& values, int num_rows, int num_cols, 
                      int avg_nnz_per_row);

void csrSpMVCpuReference(const std::vector<int>& row_ptr, const std::vector<int>& col_idx,
                        const std::vector<float>& values, const std::vector<float>& x,
                        std::vector<float>& y);

float benchmarkCuSPARSE(const int* d_row_ptr, const int* d_col_idx, const float* d_values,
                       const float* d_x, float* d_y, int num_rows, int num_cols, int nnz);

// Test and benchmark functions
void testCorrectness(void (*launchFunc)(const int*, const int*, const float*, const float*, float*, int),
                    const char* kernelName, const int* d_row_ptr, const int* d_col_idx,
                    const float* d_values, const float* d_x, float* d_y, int num_rows,
                    const std::vector<float>& reference);

double measureKernelPerformance(void (*launchFunc)(const int*, const int*, const float*, const float*, float*, int),
                               const char* kernelName, const int* d_row_ptr, const int* d_col_idx,
                               const float* d_values, const float* d_x, float* d_y, 
                               int num_rows, int nnz);

void runSimpleValidationTest();
void runBenchmarkTest(int num_rows, int num_cols, int avg_nnz_per_row);

// Kernel implementations (to be included in .cu file)

// Basic CSR SpMV kernel - one thread per row
__global__ void csr_spmv_basic(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx, 
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y,
    int num_rows
);

// Warp-per-row kernel with reduction
__global__ void csr_spmv_warp_reduction(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y,
    int num_rows
);

// Cache-optimized version using __ldg
__global__ void csr_spmv_cache_optimized(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y,
    int num_rows
);

// Vectorized implementation with manual unrolling
__global__ void csr_spmv_vectorized_safe(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y,
    int num_rows
);

// Shared memory optimization
__global__ void csr_spmv_shared_memory(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y,
    int num_rows
);

// Prefetching version
__global__ void csr_spmv_prefetch(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y,
    int num_rows
);

// Adaptive kernel
__global__ void csr_spmv_adaptive(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y,
    int num_rows
);

// Ultimate optimization kernel
__global__ void csr_spmv_ultimate(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y,
    int num_rows
);

// Simple benchmark class
class SimpleBenchmark {
private:
    cudaEvent_t start, stop;
    
public:
    SimpleBenchmark();
    ~SimpleBenchmark();
    float timeFunction(void (*func)(), int num_runs = 100);
};

// Auto-tuner class
class SpMVAutoTuner {
public:
    struct KernelProfile {
        const char* name;
        void (*func)();
        float best_time;
        bool is_best_for_size[3];  // small, medium, large
    };
    
    void profileKernels(int num_rows, int nnz);
};