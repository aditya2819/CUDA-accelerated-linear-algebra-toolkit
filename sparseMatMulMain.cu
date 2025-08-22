// sparseMatMulMain.cu - Main driver for CUDA CSR Sparse Matrix-Vector Multiplication benchmarking
#include "sparseMatMul.cuh"

int main() {
    // Display program header and information
    cout << "=== CUDA CSR Sparse Matrix-Vector Multiplication Optimization ===" << endl;
    cout << "Performance comparison of various SpMV optimization techniques" << endl << endl;
    
    // Initialize CUDA and display device information
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        cerr << "No CUDA devices found!" << endl;
        return -1;
    }
    
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    cout << "GPU: " << deviceProp.name << endl;
    cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << endl;
    cout << "Global Memory: " << (deviceProp.totalGlobalMem / (1024*1024)) << " MB" << endl;
    cout << "Shared Memory per Block: " << (deviceProp.sharedMemPerBlock / 1024) << " KB" << endl;
    cout << "Warp Size: " << deviceProp.warpSize << endl;
    cout << "Max Threads per Block: " << deviceProp.maxThreadsPerBlock << endl << endl;
    
    // Run simple validation test first to ensure correctness
    cout << "=== Correctness Verification ===" << endl;
    runSimpleValidationTest();
    
    // Define test matrix configurations
    struct TestConfig {
        int rows, cols, avg_nnz;
        const char* description;
    };
    
    TestConfig configs[] = {
        {500, 500, 10, "Small sparse matrix"},
        {2000, 2000, 15, "Medium sparse matrix"},  
        {5000, 5000, 20, "Large sparse matrix"}
    };
    
    cout << "\n=== Performance Benchmarking ===" << endl;
    
    // Results storage for analysis
    vector<pair<string, vector<double>>> all_results; // kernel_name -> [gflops for each config]
    vector<string> config_names;
    
    // Run benchmark tests with increasing matrix sizes
    for (const auto& config : configs) {
        cout << "\n" << string(60, '=') << endl;
        cout << "Testing: " << config.description << " (" << config.rows << "x" << config.cols 
             << ", avg " << config.avg_nnz << " nnz/row)" << endl;
        cout << string(60, '=') << endl;
        
        config_names.push_back(to_string(config.rows));
        
        runBenchmarkTest(config.rows, config.cols, config.avg_nnz);
        
        // Auto-tuner recommendations
        SpMVAutoTuner tuner;
        int estimated_nnz = config.rows * config.avg_nnz;
        tuner.profileKernels(config.rows, estimated_nnz);
    }
    
    // Performance summary and analysis
    cout << "\n" << string(80, '=') << endl;
    cout << "=== PERFORMANCE ANALYSIS SUMMARY ===" << endl;
    cout << string(80, '=') << endl;
    
    cout << "\nKey Optimization Techniques Tested:" << endl;
    cout << "1. Basic           - Simple one-thread-per-row approach" << endl;
    cout << "2. Warp-reduction  - Warp-level cooperation for long rows" << endl;
    cout << "3. Cache-optimized - Read-only cache hints (__ldg)" << endl;
    cout << "4. Vectorized-safe - Manual loop unrolling for ILP" << endl;
    cout << "5. Shared-memory   - Shared memory for frequently accessed data" << endl;
    cout << "6. Prefetch        - Data prefetching and aggressive unrolling" << endl;
    cout << "7. Adaptive        - Row-length aware processing strategy" << endl;
    cout << "8. Ultimate        - Combined optimizations with warp efficiency" << endl;
    
    cout << "\nOptimization Insights:" << endl;
    cout << "Memory bandwidth is often the bottleneck for sparse matrices" << endl;
    cout << "Warp-level cooperation helps with irregular sparsity patterns" << endl;
    cout << "Cache optimizations (__ldg) provide consistent improvements" << endl;
    cout << "Vectorization benefits depend on row length distribution" << endl;
    cout << "Shared memory helps when there's significant reuse in the vector" << endl;
    cout << "Adaptive strategies are crucial for mixed sparsity patterns" << endl;
    
    cout << "Benchmark completed successfully!" << endl;
    cout << "Compare your results with cuSPARSE to evaluate optimization effectiveness." << endl;
    
    return 0;
}