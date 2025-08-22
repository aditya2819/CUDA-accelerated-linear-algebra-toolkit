#ifndef CONJUGATE_GRADIENT_CUH
#define CONJUGATE_GRADIENT_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cmath>

using namespace std;
using namespace std::chrono;

// ======================== KERNEL DECLARATIONS ========================

__global__ void vectorDotProduct(const float* a, const float* b, float* result, int n);
__global__ void vectorAxpy(float alpha, const float* x, float* y, int n);
__global__ void vectorScale(float alpha, float* x, int n);

// ======================== UTILITY FUNCTION DECLARATIONS ========================

void generateTestMatrix(std::vector<int>& rowPtr, std::vector<int>& colIdx, std::vector<float>& values, int n);

// ======================== CONJUGATE GRADIENT SOLVER CLASS ========================

class ConjugateGradientSolver {
private:
    int* d_rowPtr;
    int* d_colIdx;
    float* d_values;
    int numRows, nnz;
    float* d_x, *d_r, *d_p, *d_Ap, *d_b;
    float* d_temp_scalar;
    cublasHandle_t cublasHandle;
    cusparseHandle_t cusparseHandle;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecP, vecAp;
    void* cuSparseBuffer;
    size_t cuSparseBufferSize;
    bool useCustomImplementation;

public:
    ConjugateGradientSolver(const std::vector<int>& h_rowPtr, const std::vector<int>& h_colIdx, const std::vector<float>& h_values, int n, bool useCustom = false);
    ~ConjugateGradientSolver();
    float computeDotProduct(const float* a, const float* b);
    void computeAxpy(float alpha, const float* x, float* y);
    void scaleVector(float alpha, float* x);
    void updateCuSparseVectors(); // Helper to update cuSPARSE vector descriptors
    bool solveSystem(const std::vector<float>& h_b, std::vector<float>& h_x, float tolerance = 1e-8, int maxIterations = 1000, bool verbose = false);
};

// ======================== EXTERNAL FUNCTION DECLARATIONS ========================

void launchWarpReduction(float* d_values, float* d_x, float* d_y, int* d_rowPtr, int* d_colIdx, int numRows, float alpha, float beta);

#endif