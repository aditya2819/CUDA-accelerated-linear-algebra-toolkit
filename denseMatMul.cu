#include "denseMatMul.cuh"

using namespace std;
using namespace std::chrono;

__global__ void matMulNaive(const float* A, const float* B, float* C, int N, float alpha, float beta){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

__global__ void matMulCoalesced(const float* A, const float* B, float* C, int N, float alpha, float beta){
    int row = blockIdx.x * BLOCK + (threadIdx.x / BLOCK);
    int col = blockIdx.y * BLOCK + (threadIdx.x % BLOCK);
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

__global__ void matMulTiledSharedMemory(const float* A, const float* B, float* C, int N, float alpha, float beta){
    __shared__ float sA[BLOCK][BLOCK];
    __shared__ float sB[BLOCK][BLOCK];
    int row = blockIdx.y * BLOCK + threadIdx.y;
    int col = blockIdx.x * BLOCK + threadIdx.x;
    float sum = 0.0f;
    for (int tile = 0; tile < (N + BLOCK - 1) / BLOCK; ++tile) {
        int tileCol = tile * BLOCK + threadIdx.x;
        int tileRow = tile * BLOCK + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < N && tileCol < N) ? A[row * N + tileCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (tileRow < N && col < N) ? B[tileRow * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < BLOCK; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < N && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void matMul2DTiledOptimized(const float* A, const float* B, float* C, int N, float alpha, float beta){
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;
    const uint threadRow = threadIdx.x / (BN / TN);
    const uint threadCol = threadIdx.x % (BN / TN);
    __shared__ float sA[BM * BK];
    __shared__ float sB[BK * BN];
    A += cRow * BM * N;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColA = threadIdx.x % BK;
    const uint innerRowB = threadIdx.x / BN;
    const uint innerColB = threadIdx.x % BN;
    const uint strideA = blockDim.x / BK;
    const uint strideB = blockDim.x / BN;
    float tmp[TM * TN] = {0.0};
    for (uint bkIdx = 0; bkIdx < N; bkIdx += BK) {
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            uint loadRowA = innerRowA + loadOffset;
            if (loadRowA < BM && bkIdx + innerColA < N) {
                sA[loadRowA * BK + innerColA] = A[loadRowA * N + bkIdx + innerColA];
            }
        }
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            uint loadRowB = innerRowB + loadOffset;
            if (bkIdx + loadRowB < N && innerColB < BN) {
                sB[loadRowB * BN + innerColB] = B[(bkIdx + loadRowB) * N + innerColB];
            }
        }
        __syncthreads();
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            float regM[TM], regN[TN];
            for (uint i = 0; i < TM; ++i) {
                regM[i] = sA[(threadRow * TM + i) * BK + dotIdx];
            }
            for (uint i = 0; i < TN; ++i) {
                regN[i] = sB[dotIdx * BN + threadCol * TN + i];
            }
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    tmp[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            uint globalRow = cRow * BM + threadRow * TM + resIdxM;
            uint globalCol = cCol * BN + threadCol * TN + resIdxN;
            if (globalRow < N && globalCol < N) {
                uint idx = (threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN;
                C[idx] = alpha * tmp[resIdxM * TN + resIdxN] + beta * C[idx];
            }
        }
    }
}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void matMulVectorizedFixed(const float* A, const float* B, float* C, int N, float alpha, float beta){
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;
    const uint threadRow = threadIdx.x / (BN / TN);
    const uint threadCol = threadIdx.x % (BN / TN);
    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];
    A += cRow * BM * N;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;
    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    float tmp[TM * TN] = {0.0};
    for (uint bkIdx = 0; bkIdx < N; bkIdx += BK) {
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += blockDim.x / (BK / 4)) {
            uint loadRowA = innerRowA + loadOffset;
            if (loadRowA < BM && innerColA * 4 + 3 < BK && bkIdx + innerColA * 4 + 3 < N) {
                float4 vecA = reinterpret_cast<const float4*>(&A[loadRowA * N + bkIdx + innerColA * 4])[0];
                As[(innerColA * 4 + 0) * BM + loadRowA] = vecA.x;
                As[(innerColA * 4 + 1) * BM + loadRowA] = vecA.y;
                As[(innerColA * 4 + 2) * BM + loadRowA] = vecA.z;
                As[(innerColA * 4 + 3) * BM + loadRowA] = vecA.w;
            }
        }
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += blockDim.x / (BN / 4)) {
            uint loadRowB = innerRowB + loadOffset;
            if (loadRowB < BK && innerColB * 4 + 3 < BN && bkIdx + loadRowB < N) {
                reinterpret_cast<float4*>(&Bs[loadRowB * BN + innerColB * 4])[0] =
                    reinterpret_cast<const float4*>(&B[(bkIdx + loadRowB) * N + innerColB * 4])[0];
            }
        }
        __syncthreads();
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            float regM[TM], regN[TN];
            for (uint i = 0; i < TM; ++i) {
                regM[i] = As[dotIdx * BM + threadRow * TM + i];
            }
            for (uint i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    tmp[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
            uint globalRow = cRow * BM + threadRow * TM + resIdxM;
            uint globalCol = cCol * BN + threadCol * TN + resIdxN;
            if (globalRow < N && globalCol + 3 < N && resIdxN + 3 < TN) {
                uint idx = (threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN;
                float4 original = reinterpret_cast<float4*>(&C[idx])[0];
                float4 result;
                result.x = alpha * tmp[resIdxM * TN + resIdxN + 0] + beta * original.x;
                result.y = alpha * tmp[resIdxM * TN + resIdxN + 1] + beta * original.y;
                result.z = alpha * tmp[resIdxM * TN + resIdxN + 2] + beta * original.z;
                result.w = alpha * tmp[resIdxM * TN + resIdxN + 3] + beta * original.w;
                reinterpret_cast<float4*>(&C[idx])[0] = result;
            } else {
                for (uint i = 0; i < 4 && resIdxN + i < TN; ++i) {
                    if (globalRow < N && globalCol + i < N) {
                        uint idx = (threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN + i;
                        C[idx] = alpha * tmp[resIdxM * TN + resIdxN + i] + beta * C[idx];
                    }
                }
            }
        }
    }
}

template <const int WM, const int WN, const int WK, const int WMITER, const int WNITER, const int TM, const int TN>
__global__ void matMulWarpTiled(const float* A, const float* B, float* C, int N, float alpha, float beta) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;
    const uint warpIdx = threadIdx.x / WARP_SIZE;
    const uint laneIdx = threadIdx.x % WARP_SIZE;
    const uint warpRow = warpIdx * WM;
    const uint threadRowInWarp = laneIdx / (WN / TN);
    const uint threadColInWarp = laneIdx % (WN / TN);
    const uint globalRow = cRow * WM * (blockDim.x / WARP_SIZE) + warpRow + threadRowInWarp * TM;
    const uint globalCol = cCol * WN + threadColInWarp * TN;
    __shared__ float sA[4 * WM * WK];
    __shared__ float sB[4 * WK * WN];
    float* warpSA = &sA[warpIdx * WM * WK];
    float* warpSB = &sB[warpIdx * WK * WN];
    float regC[TM * TN] = {0.0f};
    for (uint bkIdx = 0; bkIdx < N; bkIdx += WK) {
        for (uint loadIdx = laneIdx; loadIdx < WM * WK; loadIdx += WARP_SIZE) {
            uint loadRow = loadIdx / WK;
            uint loadCol = loadIdx % WK;
            uint globalLoadRow = cRow * WM * (blockDim.x / WARP_SIZE) + warpRow + loadRow;
            uint globalLoadCol = bkIdx + loadCol;
            if (globalLoadRow < N && globalLoadCol < N) {
                warpSA[loadIdx] = A[globalLoadRow * N + globalLoadCol];
            } else {
                warpSA[loadIdx] = 0.0f;
            }
        }
        for (uint loadIdx = laneIdx; loadIdx < WK * WN; loadIdx += WARP_SIZE) {
            uint loadRow = loadIdx / WN;
            uint loadCol = loadIdx % WN;
            uint globalLoadRow = bkIdx + loadRow;
            uint globalLoadCol = cCol * WN + loadCol;
            if (globalLoadRow < N && globalLoadCol < N) {
                warpSB[loadIdx] = B[globalLoadRow * N + globalLoadCol];
            } else {
                warpSB[loadIdx] = 0.0f;
            }
        }
        __syncwarp();
        for (uint dotIdx = 0; dotIdx < WK; ++dotIdx) {
            float regA[TM], regB[TN];
            for (uint i = 0; i < TM; ++i) {
                regA[i] = warpSA[(threadRowInWarp * TM + i) * WK + dotIdx];
            }
            for (uint i = 0; i < TN; ++i) {
                regB[i] = warpSB[dotIdx * WN + threadColInWarp * TN + i];
            }
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    regC[resIdxM * TN + resIdxN] += regA[resIdxM] * regB[resIdxN];
                }
            }
        }
        __syncwarp();
    }
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
            uint storeRow = globalRow + resIdxM;
            uint storeCol = globalCol + resIdxN;
            if (storeRow < N && storeCol + 3 < N && resIdxN + 3 < TN) {
                uint idx = storeRow * N + storeCol;
                float4 original = reinterpret_cast<float4*>(&C[idx])[0];
                float4 result;
                result.x = alpha * regC[resIdxM * TN + resIdxN + 0] + beta * original.x;
                result.y = alpha * regC[resIdxM * TN + resIdxN + 1] + beta * original.y;
                result.z = alpha * regC[resIdxM * TN + resIdxN + 2] + beta * original.z;
                result.w = alpha * regC[resIdxM * TN + resIdxN + 3] + beta * original.w;
                reinterpret_cast<float4*>(&C[idx])[0] = result;
            } else {
                for (uint i = 0; i < 4 && resIdxN + i < TN; ++i) {
                    if (storeRow < N && storeCol + i < N) {
                        uint idx = storeRow * N + storeCol + i;
                        C[idx] = alpha * regC[resIdxM * TN + resIdxN + i] + beta * C[idx];
                    }
                }
            }
        }
    }
}

double measureKernelPerformance(void (*setupKernel)(float*, float*, float*, int, float, float), 
                               const char* kernelName, 
                               float* d_A, float* d_B, float* d_C, int N, 
                               float alpha, float beta,
                               int numRuns) {
    vector<double> times;
    setupKernel(d_A, d_B, d_C, N, alpha, beta);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "CUDA Error in " << kernelName << ": " << cudaGetErrorString(err) << endl;
        return -1.0;
    }
    for (int i = 0; i < numRuns; ++i) {
        cudaMemset(d_C, 0, N * N * sizeof(float));
        auto start = high_resolution_clock::now();
        setupKernel(d_A, d_B, d_C, N, alpha, beta);
        cudaDeviceSynchronize();
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<nanoseconds>(end - start);
        times.push_back(duration.count() / 1e9);
    }
    double avg_time = 0.0;
    for (double t : times) avg_time += t;
    avg_time /= times.size();
    double min_time = *min_element(times.begin(), times.end());
    double gflops = (2.0 * N * N * N) / avg_time / 1e9;
    cout << kernelName << ":" << endl;
    cout << "  Time: " << fixed << setprecision(3) << avg_time * 1000 << " ms (min: " << min_time * 1000 << " ms)" << endl;
    cout << "  Performance: " << setprecision(1) << gflops << " GFLOPS" << endl;
    return gflops;
}

bool testCorrectness(void (*setupKernel)(float*, float*, float*, int, float, float),
                    const char* kernelName,
                    float* d_A, float* d_B, float* d_C, int N,
                    const vector<float>& cpu_result,
                    float alpha, float beta) {
    vector<float> h_C_init(N * N);
    for (int i = 0; i < N * N; ++i) {
        h_C_init[i] = 0.5f;
    }
    cudaMemcpy(d_C, h_C_init.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
    
    setupKernel(d_A, d_B, d_C, N, alpha, beta);
    vector<float> gpu_result(N * N);
    cudaMemcpy(gpu_result.data(), d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    int errors = 0;
    for (int i = 0; i < min(1000, N * N); ++i) {
        float expected = alpha * cpu_result[i] + beta * h_C_init[i];
        if (abs(gpu_result[i] - expected) > 0.01f) {
            errors++;
        }
    }
    bool correct = errors == 0;
    cout << kernelName << ": " << (correct ? "Correct" : "Failed") << endl;
    return correct;
}

void launchNaive(float* d_A, float* d_B, float* d_C, int N, float alpha, float beta) {
    dim3 blockSize(BLOCK, BLOCK);
    dim3 gridSize((N + BLOCK - 1) / BLOCK, (N + BLOCK - 1) / BLOCK);
    matMulNaive<<<gridSize, blockSize>>>(d_A, d_B, d_C, N, alpha, beta);
}

void launchCoalesced(float* d_A, float* d_B, float* d_C, int N, float alpha, float beta) {
    dim3 blockSize(BLOCK * BLOCK);
    dim3 gridSize((N + BLOCK - 1) / BLOCK, (N + BLOCK - 1) / BLOCK);
    matMulCoalesced<<<gridSize, blockSize>>>(d_A, d_B, d_C, N, alpha, beta);
}

void launchTiledShared(float* d_A, float* d_B, float* d_C, int N, float alpha, float beta) {
    dim3 blockSize(BLOCK, BLOCK);
    dim3 gridSize((N + BLOCK - 1) / BLOCK, (N + BLOCK - 1) / BLOCK);
    matMulTiledSharedMemory<<<gridSize, blockSize>>>(d_A, d_B, d_C, N, alpha, beta);
}

void launch2DTiledOptimized(float* d_A, float* d_B, float* d_C, int N, float alpha, float beta) {
    const int BM = 128, BN = 128, BK = 16, TM = 8, TN = 8;
    dim3 blockSize(256);
    dim3 gridSize((N + BN - 1) / BN, (N + BM - 1) / BM);
    matMul2DTiledOptimized<BM, BN, BK, TM, TN><<<gridSize, blockSize>>>(d_A, d_B, d_C, N, alpha, beta);
}

void launchVectorizedFixed(float* d_A, float* d_B, float* d_C, int N, float alpha, float beta) {
    const int BM = 128, BN = 128, BK = 16, TM = 8, TN = 8;
    dim3 blockSize(256);
    dim3 gridSize((N + BN - 1) / BN, (N + BM - 1) / BM);
    matMulVectorizedFixed<BM, BN, BK, TM, TN><<<gridSize, blockSize>>>(d_A, d_B, d_C, N, alpha, beta);
}

void launchWarpTiled(float* d_A, float* d_B, float* d_C, int N, float alpha, float beta) {
    const int WM = 64, WN = 64, WK = 16, WMITER = 2, WNITER = 2, TM = 4, TN = 4;
    const int WARPS_PER_BLOCK = 4;
    dim3 blockSize(WARPS_PER_BLOCK * WARP_SIZE);
    dim3 gridSize((N + WN - 1) / WN, (N + WM * WARPS_PER_BLOCK - 1) / (WM * WARPS_PER_BLOCK));
    matMulWarpTiled<WM, WN, WK, WMITER, WNITER, TM, TN><<<gridSize, blockSize>>>(d_A, d_B, d_C, N, alpha, beta);
}

void cpuMatMul(const vector<float>& A, const vector<float>& B, vector<float>& C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

double benchmarkCuBLAS(float* d_A, float* d_B, float* d_C, int N, float alpha, float beta, int numRuns) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    vector<double> times;
    vector<float> h_C_init(N * N, 0.5f);
    cudaMemcpy(d_C, h_C_init.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
    
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < numRuns; ++i) {
        cudaMemcpy(d_C, h_C_init.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
        auto start = high_resolution_clock::now();
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
        cudaDeviceSynchronize();
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<nanoseconds>(end - start);
        times.push_back(duration.count() / 1e9);
    }
    cublasDestroy(handle);
    double avg_time = 0.0;
    for (double t : times) avg_time += t;
    avg_time /= times.size();
    double min_time = *min_element(times.begin(), times.end());
    double gflops = (2.0 * N * N * N) / avg_time / 1e9;
    cout << "cuBLAS:" << endl;
    cout << "  Time: " << fixed << setprecision(3) << avg_time * 1000 << " ms (min: " << min_time * 1000 << " ms)" << endl;
    cout << "  Performance: " << setprecision(1) << gflops << " GFLOPS" << endl;
    return gflops;
}