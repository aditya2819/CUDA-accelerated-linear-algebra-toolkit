#include "sparseMatMul.cuh"

__global__ void csr_spmv_basic(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx, 
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y,
    int num_rows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float sum = 0.0f;
        int start = row_ptr[row], end = row_ptr[row + 1];
        for (int j = start; j < end; j++) {
            sum += values[j] * x[col_idx[j]];
        }
        y[row] = sum;
    }
}

__global__ void csr_spmv_warp_reduction(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y,
    int num_rows
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (warp_id < num_rows) {
        int row = warp_id, start = row_ptr[row], end = row_ptr[row + 1];
        float sum = 0.0f;
        for (int j = start + lane; j < end; j += 32) {
            sum += values[j] * x[col_idx[j]];
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) {
            y[row] = sum;
        }
    }
}

__global__ void csr_spmv_cache_optimized(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y,
    int num_rows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        int start = __ldg(&row_ptr[row]), end = __ldg(&row_ptr[row + 1]);
        float sum = 0.0f;
        for (int j = start; j < end; j++) {
            float val = __ldg(&values[j]);
            int col = __ldg(&col_idx[j]);
            float x_val = __ldg(&x[col]);
            sum += val * x_val;
        }
        y[row] = sum;
    }
}

__global__ void csr_spmv_vectorized_safe(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y,
    int num_rows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        int start = row_ptr[row], end = row_ptr[row + 1], j = start;
        float sum = 0.0f;
        for (; j + 3 < end; j += 4) {
            float val0 = __ldg(&values[j]), val1 = __ldg(&values[j + 1]), val2 = __ldg(&values[j + 2]), val3 = __ldg(&values[j + 3]);
            int col0 = __ldg(&col_idx[j]), col1 = __ldg(&col_idx[j + 1]), col2 = __ldg(&col_idx[j + 2]), col3 = __ldg(&col_idx[j + 3]);
            float x0 = __ldg(&x[col0]), x1 = __ldg(&x[col1]), x2 = __ldg(&x[col2]), x3 = __ldg(&x[col3]);
            sum += val0 * x0 + val1 * x1 + val2 * x2 + val3 * x3;
        }
        for (; j < end; j++) {
            sum += __ldg(&values[j]) * __ldg(&x[__ldg(&col_idx[j])]);
        }
        y[row] = sum;
    }
}

__global__ void csr_spmv_shared_memory(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y,
    int num_rows
) {
    extern __shared__ float shared_x[];
    int row = blockIdx.x * blockDim.x + threadIdx.x, tid = threadIdx.x, block_size = blockDim.x;
    for (int i = tid; i < min(1024, num_rows); i += block_size) {
        if (i < num_rows) {
            shared_x[i] = __ldg(&x[i]);
        }
    }
    __syncthreads();
    if (row < num_rows) {
        float sum = 0.0f;
        int start = __ldg(&row_ptr[row]), end = __ldg(&row_ptr[row + 1]);
        for (int j = start; j < end; j++) {
            float val = __ldg(&values[j]);
            int col = __ldg(&col_idx[j]);
            float x_val = (col < 1024) ? shared_x[col] : __ldg(&x[col]);
            sum += val * x_val;
        }
        y[row] = sum;
    }
}

__global__ void csr_spmv_prefetch(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y,
    int num_rows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        int start = __ldg(&row_ptr[row]), end = __ldg(&row_ptr[row + 1]);
        if (row + 1 < num_rows) {
            int next_start = __ldg(&row_ptr[row + 1]);
            if (next_start < end + 4) {
                __ldg(&values[next_start]);
                __ldg(&col_idx[next_start]);
            }
        }
        float sum = 0.0f;
        int j = start;
        for (; j + 7 < end; j += 8) {
            sum += __ldg(&values[j]) * __ldg(&x[__ldg(&col_idx[j])]);
            sum += __ldg(&values[j+1]) * __ldg(&x[__ldg(&col_idx[j+1])]);
            sum += __ldg(&values[j+2]) * __ldg(&x[__ldg(&col_idx[j+2])]);
            sum += __ldg(&values[j+3]) * __ldg(&x[__ldg(&col_idx[j+3])]);
            sum += __ldg(&values[j+4]) * __ldg(&x[__ldg(&col_idx[j+4])]);
            sum += __ldg(&values[j+5]) * __ldg(&x[__ldg(&col_idx[j+5])]);
            sum += __ldg(&values[j+6]) * __ldg(&x[__ldg(&col_idx[j+6])]);
            sum += __ldg(&values[j+7]) * __ldg(&x[__ldg(&col_idx[j+7])]);
        }
        for (; j < end; j++) {
            sum += __ldg(&values[j]) * __ldg(&x[__ldg(&col_idx[j])]);
        }
        y[row] = sum;
    }
}

__global__ void csr_spmv_adaptive(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y,
    int num_rows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x, lane = threadIdx.x % 32;
    if (row < num_rows) {
        int start = __ldg(&row_ptr[row]), end = __ldg(&row_ptr[row + 1]), length = end - start;
        float sum = 0.0f;
        if (length <= 4) {
            for (int j = start; j < end; j++) {
                sum += __ldg(&values[j]) * __ldg(&x[__ldg(&col_idx[j])]);
            }
        } else if (length <= 32) {
            int j = start;
            for (; j + 3 < end; j += 4) {
                float val0 = __ldg(&values[j]), val1 = __ldg(&values[j + 1]), val2 = __ldg(&values[j + 2]), val3 = __ldg(&values[j + 3]);
                int col0 = __ldg(&col_idx[j]), col1 = __ldg(&col_idx[j + 1]), col2 = __ldg(&col_idx[j + 2]), col3 = __ldg(&col_idx[j + 3]);
                sum += val0 * __ldg(&x[col0]) + val1 * __ldg(&x[col1]) + val2 * __ldg(&x[col2]) + val3 * __ldg(&x[col3]);
            }
            for (; j < end; j++) {
                sum += __ldg(&values[j]) * __ldg(&x[__ldg(&col_idx[j])]);
            }
        } else {
            for (int j = start + lane; j < end; j += 32) {
                sum += __ldg(&values[j]) * __ldg(&x[__ldg(&col_idx[j])]);
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            if (lane == 0) {
                y[row] = sum;
                return;
            } else {
                return;
            }
        }
        y[row] = sum;
    }
}

__global__ void csr_spmv_ultimate(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y,
    int num_rows
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x, warp_id = tid / 32, lane = tid % 32;
    for (int row = warp_id; row < num_rows; row += gridDim.x * blockDim.x / 32) {
        int start = __ldg(&row_ptr[row]), end = __ldg(&row_ptr[row + 1]), length = end - start;
        float sum = 0.0f;
        if (length <= 8) {
            if (lane < length) {
                int j = start + lane;
                sum = __ldg(&values[j]) * __ldg(&x[__ldg(&col_idx[j])]);
            }
            #pragma unroll
            for (int offset = 4; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xff, sum, offset);
            }
            if (lane == 0) {
                y[row] = sum;
            }
        } else if (length <= 64) {
            for (int j = start + lane; j < end; j += 32) {
                sum += __ldg(&values[j]) * __ldg(&x[__ldg(&col_idx[j])]);
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            if (lane == 0) {
                y[row] = sum;
            }
        } else {
            for (int j = start + lane; j < end; j += 32) {
                float val = __ldg(&values[j]);
                int col = __ldg(&col_idx[j]);
                sum += val * __ldg(&x[col]);
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            if (lane == 0) {
                y[row] = sum;
            }
        }
    }
}

void launchBasicSpMV(const int* d_row_ptr, const int* d_col_idx, const float* d_values, const float* d_x, float* d_y, int num_rows) {
    dim3 block(256), grid((num_rows + block.x - 1) / block.x);
    csr_spmv_basic<<<grid, block>>>(d_row_ptr, d_col_idx, d_values, d_x, d_y, num_rows);
}

void launchWarpReductionSpMV(const int* d_row_ptr, const int* d_col_idx, const float* d_values, const float* d_x, float* d_y, int num_rows) {
    dim3 block(256), grid(((num_rows * 32) + block.x - 1) / block.x);
    csr_spmv_warp_reduction<<<grid, block>>>(d_row_ptr, d_col_idx, d_values, d_x, d_y, num_rows);
}

void launchCacheOptimizedSpMV(const int* d_row_ptr, const int* d_col_idx, const float* d_values, const float* d_x, float* d_y, int num_rows) {
    dim3 block(256), grid((num_rows + block.x - 1) / block.x);
    csr_spmv_cache_optimized<<<grid, block>>>(d_row_ptr, d_col_idx, d_values, d_x, d_y, num_rows);
}

void launchVectorizedSpMV(const int* d_row_ptr, const int* d_col_idx, const float* d_values, const float* d_x, float* d_y, int num_rows) {
    dim3 block(256), grid((num_rows + block.x - 1) / block.x);
    csr_spmv_vectorized_safe<<<grid, block>>>(d_row_ptr, d_col_idx, d_values, d_x, d_y, num_rows);
}

void launchSharedMemorySpMV(const int* d_row_ptr, const int* d_col_idx, const float* d_values, const float* d_x, float* d_y, int num_rows) {
    dim3 block(256), grid((num_rows + block.x - 1) / block.x);
    size_t shared_mem_size = 1024 * sizeof(float);
    csr_spmv_shared_memory<<<grid, block, shared_mem_size>>>(d_row_ptr, d_col_idx, d_values, d_x, d_y, num_rows);
}

void launchPrefetchSpMV(const int* d_row_ptr, const int* d_col_idx, const float* d_values, const float* d_x, float* d_y, int num_rows) {
    dim3 block(256), grid((num_rows + block.x - 1) / block.x);
    csr_spmv_prefetch<<<grid, block>>>(d_row_ptr, d_col_idx, d_values, d_x, d_y, num_rows);
}

void launchAdaptiveSpMV(const int* d_row_ptr, const int* d_col_idx, const float* d_values, const float* d_x, float* d_y, int num_rows) {
    dim3 block(256), grid((num_rows + block.x - 1) / block.x);
    csr_spmv_adaptive<<<grid, block>>>(d_row_ptr, d_col_idx, d_values, d_x, d_y, num_rows);
}

void launchUltimateSpMV(const int* d_row_ptr, const int* d_col_idx, const float* d_values, const float* d_x, float* d_y, int num_rows) {
    dim3 block(256), grid(min(65535, (num_rows + 31) / 32));
    csr_spmv_ultimate<<<grid, block>>>(d_row_ptr, d_col_idx, d_values, d_x, d_y, num_rows);
}

void generateCSRMatrix(std::vector<int>& row_ptr, std::vector<int>& col_idx, std::vector<float>& values, int num_rows, int num_cols, int avg_nnz_per_row) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> col_dis(0, num_cols - 1);
    std::uniform_real_distribution<float> val_dis(0.1f, 2.0f);
    row_ptr.resize(num_rows + 1);
    row_ptr[0] = 0;
    std::vector<std::vector<std::pair<int, float>>> temp_matrix(num_rows);
    for (int row = 0; row < num_rows; row++) {
        std::set<int> used_cols;
        int variation = (int)(gen() % 5) - 2, nnz_this_row = std::max(1, avg_nnz_per_row + variation);
        nnz_this_row = std::min(nnz_this_row, num_cols);
        for (int k = 0; k < nnz_this_row; k++) {
            int col;
            do {
                col = col_dis(gen);
            } while (used_cols.count(col));
            used_cols.insert(col);
            float val = val_dis(gen);
            temp_matrix[row].push_back(std::make_pair(col, val));
        }
        std::sort(temp_matrix[row].begin(), temp_matrix[row].end());
        row_ptr[row + 1] = row_ptr[row] + (int)temp_matrix[row].size();
    }
    int total_nnz = row_ptr[num_rows], idx = 0;
    col_idx.resize(total_nnz);
    values.resize(total_nnz);
    for (int row = 0; row < num_rows; row++) {
        for (const auto& entry : temp_matrix[row]) {
            col_idx[idx] = entry.first;
            values[idx] = entry.second;
            idx++;
        }
    }
    cout << "Generated CSR matrix: " << num_rows << "x" << num_cols << ", NNZ=" << total_nnz << endl;
}

void csrSpMVCpuReference(const std::vector<int>& row_ptr, const std::vector<int>& col_idx, const std::vector<float>& values, const std::vector<float>& x, std::vector<float>& y) {
    int num_rows = (int)row_ptr.size() - 1;
    y.assign(num_rows, 0.0f);
    for (int row = 0; row < num_rows; row++) {
        float sum = 0.0f;
        for (int j = row_ptr[row]; j < row_ptr[row + 1]; j++) {
            if (j < (int)values.size() && col_idx[j] < (int)x.size()) {
                sum += values[j] * x[col_idx[j]];
            }
        }
        y[row] = sum;
    }
}

float benchmarkCuSPARSE(const int* d_row_ptr, const int* d_col_idx, const float* d_values, const float* d_x, float* d_y, int num_rows, int num_cols, int nnz) {
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateCsr(&matA, num_rows, num_cols, nnz,
                     (void*)d_row_ptr, (void*)d_col_idx, (void*)d_values,
                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnVec(&vecX, num_cols, (void*)d_x, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, num_rows, (void*)d_y, CUDA_R_32F);
    float alpha = 1.0f, beta = 0.0f;
    size_t bufferSize;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                           CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    void* buffer;
    CUDA_CHECK(cudaMalloc(&buffer, bufferSize));
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                CUSPARSE_SPMV_ALG_DEFAULT, buffer);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    const int num_runs = 100;
    for (int i = 0; i < num_runs; i++) {
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                    CUSPARSE_SPMV_ALG_DEFAULT, buffer);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= (float)num_runs;
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroySpMat(matA);
    cusparseDestroy(handle);
    cudaFree(buffer);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

void testCorrectness(void (*launchFunc)(const int*, const int*, const float*, const float*, float*, int), const char* kernelName, const int* d_row_ptr, const int* d_col_idx, const float* d_values, const float* d_x, float* d_y, int num_rows, const std::vector<float>& reference) {
    CUDA_CHECK(cudaMemset(d_y, 0, num_rows * sizeof(float)));
    launchFunc(d_row_ptr, d_col_idx, d_values, d_x, d_y, num_rows);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> result(num_rows);
    CUDA_CHECK(cudaMemcpy(result.data(), d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost));
    bool correct = true;
    float max_error = 0.0f;
    for (int i = 0; i < num_rows; i++) {
        float error = std::abs(reference[i] - result[i]);
        max_error = std::max(max_error, error);
        if (error > 1e-1f) {
            correct = false;
            break;
        }
    }
    if (correct) {
        cout << kernelName << " PASSED (max error: " << max_error << ")" << endl;
    } else {
        cout << kernelName << " FAILED (max error: " << max_error << ")" << endl;
    }
}

double measureKernelPerformance(void (*launchFunc)(const int*, const int*, const float*, const float*, float*, int), const char* kernelName, const int* d_row_ptr, const int* d_col_idx, const float* d_values, const float* d_x, float* d_y, int num_rows, int nnz) {
    launchFunc(d_row_ptr, d_col_idx, d_values, d_x, d_y, num_rows);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    const int num_runs = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_runs; i++) {
        launchFunc(d_row_ptr, d_col_idx, d_values, d_x, d_y, num_rows);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= (float)num_runs;
    long long flops = 2LL * nnz;
    double gflops = (flops / 1e9) / (ms / 1000.0);
    cout << kernelName << ": " << setprecision(3) << ms << " ms, " 
         << setprecision(1) << gflops << " GFLOPS" << endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return gflops;
}

void runSimpleValidationTest() {
    std::vector<int> row_ptr = {0, 2, 4, 6, 8}, col_idx = {0, 2, 1, 3, 0, 3, 1, 2};
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, x = {1.0f, 2.0f, 3.0f, 4.0f};
    int num_rows = 4, num_cols = 4, nnz = 8;
    std::vector<float> y_cpu;
    csrSpMVCpuReference(row_ptr, col_idx, values, x, y_cpu);
    cout << "Expected: [7, 22, 29, 38]" << endl;
    cout << "CPU got:  [";
    for (int i = 0; i < num_rows; i++) {
        cout << y_cpu[i];
        if (i < num_rows - 1) cout << ", ";
    }
    cout << "]" << endl;
    int *d_row_ptr, *d_col_idx;
    float *d_values, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (num_rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, num_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, num_rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr.data(), (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, col_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), num_cols * sizeof(float), cudaMemcpyHostToDevice));
    testCorrectness(launchBasicSpMV, "Basic SpMV", d_row_ptr, d_col_idx, d_values, d_x, d_y, num_rows, y_cpu);
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
}

void runBenchmarkTest(int num_rows, int num_cols, int avg_nnz_per_row) {
    std::vector<int> h_row_ptr, h_col_idx;
    std::vector<float> h_values;
    generateCSRMatrix(h_row_ptr, h_col_idx, h_values, num_rows, num_cols, avg_nnz_per_row);
    int nnz = (int)h_values.size();
    std::vector<float> h_x(num_cols);
    for (int i = 0; i < num_cols; i++) {
        h_x[i] = (float)(i + 1);
    }
    std::vector<float> y_cpu;
    csrSpMVCpuReference(h_row_ptr, h_col_idx, h_values, h_x, y_cpu);
    int *d_row_ptr, *d_col_idx;
    float *d_values, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (num_rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, num_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, num_rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, h_row_ptr.data(), (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, h_col_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), num_cols * sizeof(float), cudaMemcpyHostToDevice));
    long long flops = 2LL * nnz;
    long long bytes = (long long)nnz * (sizeof(float) + sizeof(int)) + (long long)num_cols * sizeof(float) + (long long)num_rows * sizeof(float);
    float cusparse_time = benchmarkCuSPARSE(d_row_ptr, d_col_idx, d_values, d_x, d_y, num_rows, num_cols, nnz);
    cout << "\nResults:" << endl;
    cout << left << setw(20) << "Kernel" << setw(12) << "Time(ms)" << setw(12) << "GFLOPS" << setw(15) << "BW(GB/s)" << setw(10) << "Speedup" << endl;
    cout << string(70, '-') << endl;
    cout << fixed << setprecision(3);
    cout << setw(20) << "cuSPARSE" << setw(12) << cusparse_time << setw(12) << (flops / 1e9f) / (cusparse_time / 1000.0f)
         << setw(15) << (bytes / 1e9f) / (cusparse_time / 1000.0f) << setw(10) << "1.000x" << endl;
    struct KernelInfo {
        const char* name;
        void (*func)(const int*, const int*, const float*, const float*, float*, int);
    };
    vector<KernelInfo> kernels = {
        {"Basic", launchBasicSpMV},
        {"Warp-reduction", launchWarpReductionSpMV},
        {"Cache-optimized", launchCacheOptimizedSpMV},
        {"Vectorized-safe", launchVectorizedSpMV},
        {"Shared-memory", launchSharedMemorySpMV},
        {"Prefetch", launchPrefetchSpMV},
        {"Adaptive", launchAdaptiveSpMV},
        {"Ultimate", launchUltimateSpMV}
    };
    vector<pair<string, double>> results;
    for (const auto& kernel : kernels) {
        testCorrectness(kernel.func, kernel.name, d_row_ptr, d_col_idx, d_values, d_x, d_y, num_rows, y_cpu);
        double gflops = measureKernelPerformance(kernel.func, kernel.name, d_row_ptr, d_col_idx, d_values, d_x, d_y, num_rows, nnz);
        if (gflops > 0) results.push_back({kernel.name, gflops});
        float kernel_time = (flops / 1e9f) / gflops * 1000.0f;
        cout << setw(20) << kernel.name << setw(12) << kernel_time << setw(12) << gflops
             << setw(15) << (bytes / 1e9f) / (kernel_time / 1000.0f) << setw(10) << (cusparse_time / kernel_time) << "x" << endl;
    }
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
}

SimpleBenchmark::SimpleBenchmark() {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
}

SimpleBenchmark::~SimpleBenchmark() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

float SimpleBenchmark::timeFunction(void (*func)(), int num_runs) {
    func();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_runs; i++) {
        func();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms / (float)num_runs;
}

void SpMVAutoTuner::profileKernels(int num_rows, int nnz) {
    if (num_rows <= 1000) {
        cout << "Recommendation: Use Warp-reduction kernel for matrices ≤1000 rows" << endl;
    } else if (num_rows <= 3000) {
        cout << "Recommendation: Use Prefetch kernel for medium matrices (1K-3K rows)" << endl;
    } else {
        cout << "Recommendation: Use Vectorized-safe kernel for large matrices (>3K rows)" << endl;
    }
    float avg_nnz_per_row = (float)nnz / num_rows;
    if (avg_nnz_per_row < 5) {
        cout << "Note: Very sparse matrix - consider COO format for better performance" << endl;
    } else if (avg_nnz_per_row > 50) {
        cout << "Note: Dense matrix - consider dense GEMV for better performance" << endl;
    }
}