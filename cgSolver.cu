#include "sparseMatMul.cuh"
#include "denseMatMul.cuh"
#include "cgSolver.cuh"

__global__ void vectorDotProduct(const float* a, const float* b, float* result, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (idx < n) ? a[idx] * b[idx] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) atomicAdd(result, sdata[0]);
}

__global__ void vectorAxpy(float alpha, const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] += alpha * x[idx];
    }
}

__global__ void vectorScale(float alpha, float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] *= alpha;
    }
}

void generateTestMatrix(vector<int>& rowPtr, vector<int>& colIdx, vector<float>& values, int n) {
    rowPtr.resize(n + 1);
    colIdx.clear();
    values.clear();
    rowPtr[0] = 0;
    for (int i = 0; i < n; i++) {
        colIdx.push_back(i);
        values.push_back(4.0f);
        if (i > 0) {
            colIdx.push_back(i - 1);
            values.push_back(-1.0f);
        }
        if (i < n - 1) {
            colIdx.push_back(i + 1);
            values.push_back(-1.0f);
        }
        rowPtr[i + 1] = values.size();
    }
    for (int i = 0; i < n; i++) {
        vector<pair<int, float>> row_entries;
        for (int j = rowPtr[i]; j < rowPtr[i + 1]; j++) {
            row_entries.push_back({colIdx[j], values[j]});
        }
        sort(row_entries.begin(), row_entries.end());
        for (size_t j = 0; j < row_entries.size(); j++) {
            colIdx[rowPtr[i] + j] = row_entries[j].first;
            values[rowPtr[i] + j] = row_entries[j].second;
        }
    }
}

ConjugateGradientSolver::ConjugateGradientSolver(const vector<int>& h_rowPtr, const vector<int>& h_colIdx, const vector<float>& h_values, int n, bool useCustom) 
    : numRows(n), nnz(h_values.size()), useCustomImplementation(useCustom), cuSparseBuffer(nullptr), cuSparseBufferSize(0) {
    cudaMalloc(&d_rowPtr, (numRows + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, nnz * sizeof(int));
    cudaMalloc(&d_values, nnz * sizeof(float));
    cudaMemcpy(d_rowPtr, h_rowPtr.data(), (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, h_colIdx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_x, numRows * sizeof(float));
    cudaMalloc(&d_r, numRows * sizeof(float));
    cudaMalloc(&d_p, numRows * sizeof(float));
    cudaMalloc(&d_Ap, numRows * sizeof(float));
    cudaMalloc(&d_b, numRows * sizeof(float));
    cudaMalloc(&d_temp_scalar, sizeof(float));
    cublasCreate(&cublasHandle);
    
    cusparseCreate(&cusparseHandle);
    cusparseCreateCsr(&matA, numRows, numRows, nnz,
                      d_rowPtr, d_colIdx, d_values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnVec(&vecP, numRows, d_p, CUDA_R_32F);
    cusparseCreateDnVec(&vecAp, numRows, d_Ap, CUDA_R_32F);
    const float alpha = 1.0f, beta = 0.0f;
    cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           &alpha, matA, vecP, &beta, vecAp, CUDA_R_32F,
                           CUSPARSE_SPMV_ALG_DEFAULT, &cuSparseBufferSize);
    cudaMalloc(&cuSparseBuffer, cuSparseBufferSize);
}

ConjugateGradientSolver::~ConjugateGradientSolver() {
    cudaFree(d_rowPtr); cudaFree(d_colIdx); cudaFree(d_values);
    cudaFree(d_x); cudaFree(d_r); cudaFree(d_p); cudaFree(d_Ap); cudaFree(d_b);
    cudaFree(d_temp_scalar);
    cublasDestroy(cublasHandle);
    
    if (cuSparseBuffer) cudaFree(cuSparseBuffer);
    cusparseDestroyDnVec(vecP);
    cusparseDestroyDnVec(vecAp);
    cusparseDestroySpMat(matA);
    cusparseDestroy(cusparseHandle);
}

float ConjugateGradientSolver::computeDotProduct(const float* a, const float* b) {
    cudaMemset(d_temp_scalar, 0, sizeof(float));
    dim3 block(256), grid((numRows + block.x - 1) / block.x);
    vectorDotProduct<<<grid, block, block.x * sizeof(float)>>>(a, b, d_temp_scalar, numRows);
    cudaDeviceSynchronize();
    float result;
    cudaMemcpy(&result, d_temp_scalar, sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}

void ConjugateGradientSolver::computeAxpy(float alpha, const float* x, float* y) {
    dim3 block(256), grid((numRows + block.x - 1) / block.x);
    vectorAxpy<<<grid, block>>>(alpha, x, y, numRows);
    cudaDeviceSynchronize();
}

void ConjugateGradientSolver::scaleVector(float alpha, float* x) {
    dim3 block(256), grid((numRows + block.x - 1) / block.x);
    vectorScale<<<grid, block>>>(alpha, x, numRows);
    cudaDeviceSynchronize();
}

void ConjugateGradientSolver::updateCuSparseVectors() {
    cusparseDestroyDnVec(vecP);
    cusparseDestroyDnVec(vecAp);
    cusparseCreateDnVec(&vecP, numRows, d_p, CUDA_R_32F);
    cusparseCreateDnVec(&vecAp, numRows, d_Ap, CUDA_R_32F);
    const float alpha = 1.0f, beta = 0.0f;
    size_t newBufferSize;
    cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           &alpha, matA, vecP, &beta, vecAp, CUDA_R_32F,
                           CUSPARSE_SPMV_ALG_DEFAULT, &newBufferSize);
    if (newBufferSize > cuSparseBufferSize) {
        if (cuSparseBuffer) cudaFree(cuSparseBuffer);
        cudaMalloc(&cuSparseBuffer, newBufferSize);
        cuSparseBufferSize = newBufferSize;
    }
}

bool ConjugateGradientSolver::solveSystem(const vector<float>& h_b, vector<float>& h_x, float tolerance, int maxIterations, bool verbose) {
    cudaMemcpy(d_b, h_b.data(), numRows * sizeof(float), cudaMemcpyHostToDevice);
    const float one = 1.0f;
    cudaMemset(d_x, 0, numRows * sizeof(float));
    cudaMemcpy(d_r, d_b, numRows * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_p, d_r, numRows * sizeof(float), cudaMemcpyDeviceToDevice);
    if (!useCustomImplementation) {
        updateCuSparseVectors();
    }
    float rsold;
    if (useCustomImplementation) {
        rsold = computeDotProduct(d_r, d_r);
    } else {
        cublasSdot(cublasHandle, numRows, d_r, 1, d_r, 1, &rsold);
    }
    if (verbose) {
        cout << "  Initial residual norm: " << scientific << setprecision(3) << sqrt(rsold) << endl;
    }
    auto startTime = high_resolution_clock::now();
    for (int iteration = 0; iteration < maxIterations; iteration++) {
        if (useCustomImplementation) {
            launchAdaptiveSpMV(d_rowPtr, d_colIdx, d_values, d_p, d_Ap, numRows);
        } else {
            const float alpha = 1.0f, beta = 0.0f;
            cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, matA, vecP, &beta, vecAp, CUDA_R_32F,
                        CUSPARSE_SPMV_ALG_DEFAULT, cuSparseBuffer);
        }
        cudaDeviceSynchronize();
        float pAp;
        if (useCustomImplementation) {
            pAp = computeDotProduct(d_p, d_Ap);
        } else {
            cublasSdot(cublasHandle, numRows, d_p, 1, d_Ap, 1, &pAp);
        }
        if (pAp <= 0 || !isfinite(pAp)) {
            if (verbose) cout << "  Error: Matrix not positive definite" << endl;
            return false;
        }
        float alpha = rsold / pAp;
        if (useCustomImplementation) {
            computeAxpy(alpha, d_p, d_x);
        } else {
            cublasSaxpy(cublasHandle, numRows, &alpha, d_p, 1, d_x, 1);
        }
        if (useCustomImplementation) {
            computeAxpy(-alpha, d_Ap, d_r);
        } else {
            float negAlpha = -alpha;
            cublasSaxpy(cublasHandle, numRows, &negAlpha, d_Ap, 1, d_r, 1);
        }
        float rsnew;
        if (useCustomImplementation) {
            rsnew = computeDotProduct(d_r, d_r);
        } else {
            cublasSdot(cublasHandle, numRows, d_r, 1, d_r, 1, &rsnew);
        }
        if (verbose && (iteration % 25 == 0 || sqrt(rsnew) < tolerance)) {
            cout << "    Iteration " << setw(3) << iteration << ": residual = " 
                 << scientific << setprecision(3) << sqrt(rsnew) << endl;
        }
        if (rsnew < tolerance * tolerance) {
            auto endTime = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(endTime - startTime);
            if (verbose) {
                cout << "  Converged in " << iteration + 1 << " iterations" << endl;
                cout << "  Solve time: " << duration.count() / 1000.0 << " ms" << endl;
                cout << "  Final residual: " << scientific << sqrt(rsnew) << endl;
            }
            h_x.resize(numRows);
            cudaMemcpy(h_x.data(), d_x, numRows * sizeof(float), cudaMemcpyDeviceToHost);
            return true;
        }
        float beta = rsnew / rsold;
        if (useCustomImplementation) {
            scaleVector(beta, d_p);
            computeAxpy(1.0f, d_r, d_p);
        } else {
            cublasSscal(cublasHandle, numRows, &beta, d_p, 1);
            cublasSaxpy(cublasHandle, numRows, &one, d_r, 1, d_p, 1);
        }
        rsold = rsnew;
    }
    if (verbose) {
        cout << "  Warning: Maximum iterations reached without convergence" << endl;
    }
    return false;
}