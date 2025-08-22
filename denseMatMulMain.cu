#include "denseMatMul.cuh"

int main() {
    // Define problem configuration parameters
    const int N = 2048;                    // Matrix dimensions (N x N square matrices)
    const float alpha = 1.0f;              // Scaling factor for matrix product A*B
    const float beta = 0.0f;               // Scaling factor for existing matrix C (GEMM operation: C = α*A*B + β*C)
    
    // Display program header and configuration
    cout << "=== CUDA Matrix Multiplication with Alpha/Beta Scalars ===" << endl;
    cout << "Matrix size: " << N << "x" << N << " (" << setprecision(1) << (2.0*N*N*N/1e9) << " GFLOPs total)" << endl;
    cout << "Computing: C = " << alpha << " * A * B + " << beta << " * C" << endl << endl;
    
    // Allocate host memory for input matrices and CPU reference result
    vector<float> h_A(N * N), h_B(N * N), h_C_cpu(N * N);  // Host matrices: A, B, and CPU result
    
    // Initialize matrices with random values for realistic performance testing
    cout << "Initializing random matrices..." << endl;
    mt19937 gen(42);                       // Mersenne Twister random generator with fixed seed for reproducibility
    uniform_real_distribution<float> dis(-1.0f, 1.0f);     // Uniform distribution in range [-1, 1]
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = dis(gen);                 // Fill matrix A with random values
        h_B[i] = dis(gen);                 // Fill matrix B with random values
    }
    
    // Allocate GPU memory for matrices
    float *d_A, *d_B, *d_C;               // Device pointers for matrices A, B, and C
    size_t size = N * N * sizeof(float);   // Memory size for each matrix (N²* 4 bytes per float)
    cudaMalloc(&d_A, size);                // Allocate GPU memory for matrix A
    cudaMalloc(&d_B, size);                // Allocate GPU memory for matrix B
    cudaMalloc(&d_C, size);                // Allocate GPU memory for matrix C (result)
    
    // Transfer input matrices from host to device
    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);  // Copy A to GPU
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);  // Copy B to GPU
    
    // Compute CPU reference implementation for correctness verification
    cout << "Computing CPU reference (A*B only)..." << endl;
    auto cpu_start = high_resolution_clock::now();
    cpuMatMul(h_A, h_B, h_C_cpu, N);      // Standard triple-loop CPU matrix multiplication
    auto cpu_end = high_resolution_clock::now();
    double cpu_time = duration_cast<milliseconds>(cpu_end - cpu_start).count() / 1000.0;
    cout << "CPU time: " << setprecision(2) << cpu_time << " s" << endl << endl;
    
    // Correctness verification section - test all GPU implementations against CPU reference
    cout << "=== Correctness Tests (with alpha=" << alpha << ", beta=" << beta << ") ===" << endl;
    
    // Test 1: Naive implementation (basic thread-per-element approach)
    testCorrectness(launchNaive, "Naive", d_A, d_B, d_C, N, h_C_cpu, alpha, beta);
    
    // Test 2: Coalesced memory access optimization
    testCorrectness(launchCoalesced, "Coalesced", d_A, d_B, d_C, N, h_C_cpu, alpha, beta);
    
    // Test 3: Shared memory tiling to reduce global memory accesses
    testCorrectness(launchTiledShared, "Shared Memory", d_A, d_B, d_C, N, h_C_cpu, alpha, beta);
    
    // Test 4: Advanced 2D tiling with optimized memory patterns
    testCorrectness(launch2DTiled, "2D Tiled Optimized", d_A, d_B, d_C, N, h_C_cpu, alpha, beta);
    
    // Test 5: Vectorized operations using float4 for improved memory throughput
    testCorrectness(launchVectorized, "Vectorized Fixed", d_A, d_B, d_C, N, h_C_cpu, alpha, beta);
    
    // Test 6: Warp-level tiling for better thread cooperation and reduced divergence
    testCorrectness(launchWarpTiled, "Warp Tiled", d_A, d_B, d_C, N, h_C_cpu, alpha, beta);
    
    // Performance benchmarking section
    cout << "\n=== Performance Benchmarks (with alpha=" << alpha << ", beta=" << beta << ") ===" << endl;
    vector<pair<string, double>> results;  // Storage for performance results (name, GFLOPS)
    
    // Benchmark 1: Naive implementation
    double gflops1 = measureKernelPerformance(launchNaive, "Naive", d_A, d_B, d_C, N, alpha, beta);
    if (gflops1 > 0) results.push_back({"Naive", gflops1});
    
    // Benchmark 2: Coalesced memory access
    double gflops2 = measureKernelPerformance(launchCoalesced, "Coalesced", d_A, d_B, d_C, N, alpha, beta);
    if (gflops2 > 0) results.push_back({"Coalesced", gflops2});
    
    // Benchmark 3: Shared memory optimization
    double gflops3 = measureKernelPerformance(launchTiledShared, "Shared Memory", d_A, d_B, d_C, N, alpha, beta);
    if (gflops3 > 0) results.push_back({"Shared Memory", gflops3});
    
    // Benchmark 4: 2D tiled optimization
    double gflops4 = measureKernelPerformance(launch2DTiled, "2D Tiled Optimized", d_A, d_B, d_C, N, alpha, beta);
    if (gflops4 > 0) results.push_back({"2D Tiled Optimized", gflops4});
    
    // Benchmark 5: Vectorized implementation
    double gflops5 = measureKernelPerformance(launchVectorized, "Vectorized Fixed", d_A, d_B, d_C, N, alpha, beta);
    if (gflops5 > 0) results.push_back({"Vectorized Fixed", gflops5});
    
    // Benchmark 6: Warp-level tiling
    double gflops6 = measureKernelPerformance(launchWarpTiled, "Warp Tiled", d_A, d_B, d_C, N, alpha, beta);
    if (gflops6 > 0) results.push_back({"Warp Tiled", gflops6});
    
    // Benchmark cuBLAS (NVIDIA's highly optimized library) as reference
    cout << "\n[TEST] Testing cuBLAS (NVIDIA's optimized library) with alpha/beta..." << endl;
    double cublas_gflops = benchmarkCuBLAS(d_A, d_B, d_C, N, alpha, beta);
    
    // Performance analysis and ranking
    cout << "\n=== Final Results ===" << endl;
    
    // Find the best performing custom kernel
    auto best = max_element(results.begin(), results.end(), 
        [](const auto& a, const auto& b) { return a.second < b.second; });
    
    if (best != results.end()) {
        // Display best custom kernel performance
        cout << "[BEST] Best Custom Kernel: " << best->first << " at " << setprecision(1) << best->second << " GFLOPS" << endl;
        
        // Compare against cuBLAS performance (industry standard)
        cout << "[VS] vs cuBLAS (" << setprecision(1) << cublas_gflops << " GFLOPS): " << setprecision(1) << (best->second / cublas_gflops) * 100 << "%" << endl;
        
        // Provide performance assessment
        if (best->second > cublas_gflops * 0.8) {
            cout << "[EXCELLENT] Outstanding! Your kernel achieves " << setprecision(1) << (best->second / cublas_gflops) * 100 << "% of cuBLAS performance!" << endl;
        }
        
        // Create comprehensive performance ranking including cuBLAS
        cout << "\n=== Performance Ranking ===" << endl;
        vector<pair<string, double>> all_results = results;      // Copy custom results
        all_results.push_back({"cuBLAS", cublas_gflops});        // Add cuBLAS reference
        
        // Sort by performance (highest GFLOPS first)
        sort(all_results.begin(), all_results.end(), [](const auto& a, const auto& b) { 
            return a.second > b.second; 
        });
        
        // Display ranked results with medal indicators
        for (size_t i = 0; i < all_results.size(); ++i) {
            string medal = (i == 0) ? "[1st]" : (i == 1) ? "[2nd]" : (i == 2) ? "[3rd]" : "     ";
            cout << medal << " " << left << setw(20) << all_results[i].first 
                 << setprecision(1) << all_results[i].second << " GFLOPS" << endl;
        }
    }
    
    // Clean up GPU memory allocations
    cudaFree(d_A);                         // Free matrix A memory
    cudaFree(d_B);                         // Free matrix B memory
    cudaFree(d_C);                         // Free matrix C memory
    
    return 0;                              // Program completed successfully
}