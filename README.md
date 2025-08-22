# CUDA-Accelerated Linear Algebra Toolkit

A high-performance GPU-accelerated linear algebra toolkit implementing dense matrix operations, sparse matrix operations, and iterative linear solvers for circuit simulation workloads.

## Project Overview

This project provides a CUDA-based linear algebra library featuring:

- **Dense Matrix Operations**: Optimized tiled matrix multiplication with multiple kernel implementations
- **Sparse Matrix Operations**: CSR SpMV (Sparse matrix-vector multiplication) with custom kernels and cuSPARSE integration
- **Iterative Linear Solver**: Conjugate Gradient method with GPU preconditioning
- **Performance Benchmarking**: CPU vs GPU performance comparisons and custom vs library implementations

## Features

### Dense Matrix Multiplication (`denseMatMul`)
- Multiple optimization strategies: Naive, Coalesced, Shared Memory, 2D Tiled, Vectorized, Warp Tiled
- GEMM operations with alpha/beta scaling: `C = α*A*B + β*C`
- Performance comparison against cuBLAS
- Support for large matrices (2048×2048 and beyond)

### Sparse Matrix Operations (`sparseMatMul`)
- CSR (Compressed Sparse Row) format support
- Multiple SpMV kernel optimizations: Basic, Warp-reduction, Cache-optimized, Vectorized, Shared-memory, Prefetch, Adaptive
- Performance comparison against cuSPARSE
- Configurable sparsity patterns for different problem sizes

### Conjugate Gradient Solver (`cgSolver`)
- GPU-accelerated iterative linear solver
- Support for symmetric positive definite matrices
- Custom CUDA kernels vs cuBLAS + cuSPARSE implementation comparison
- Configurable convergence tolerance and maximum iterations

## Building the Project

### Prerequisites
- CUDA Toolkit (11.0 or later)
- CMake (3.18 or later)
- C++17 compatible compiler
- NVIDIA GPU with Compute Capability 7.0 or higher

### Build Instructions

1. **Clone and navigate to the project directory:**
   ```bash
   cd CUDA
   ```

2. **Create build directory:**
   ```bash
   mkdir build
   cd build
   ```

3. **Configure with CMake:**
   ```bash
   cmake ..
   ```

4. **Build the project:**
   ```bash
   cmake --build . --config Release
   ```

5. **Run all benchmarks:**
   ```bash
   cmake --build . --target run_benchmarks
   ```

### Alternative: Direct NVCC Compilation
```bash
# Dense matrix multiplication
nvcc -o denseMatMul.exe denseMatMulMain.cu denseMatMul.cu -lcublas

# Sparse matrix multiplication  
nvcc -o sparseMatMul.exe sparseMatMulMain.cu sparseMatMul.cu -lcusparse

# Conjugate gradient solver
nvcc -o cgSolver.exe cgSolverMain.cu cgSolver.cu sparseMatMul.cu denseMatMul.cu -lcusparse -lcublas
```

## System Specifications

*All benchmarks were performed on the following system configuration:*

### Hardware Configuration
- **GPU**: NVIDIA GeForce RTX 4070 Laptop GPU
  - Compute Capability: 8.9 (Ada Lovelace architecture)
  - VRAM: 8,187 MB (~8GB GDDR6X)
  - Shared Memory per Block: 48 KB
  - Warp Size: 32 threads
  - Max Threads per Block: 1,024
- **CPU**: Intel Core i9-14900HX (24 cores, 32 threads)
- **RAM**: 16GB DDR5
- **Platform**: Windows with CUDA Toolkit

### Software Environment
- **CUDA Toolkit**: 12.x
- **Compiler**: NVCC with Visual Studio 2022 Community
- **Build System**: CMake 3.18+ / Direct NVCC compilation
- **Libraries**: cuBLAS, cuSPARSE

## Performance Results

*Note: All benchmarks and this README were generated using Claude AI assistant*

### Dense Matrix Multiplication Results

**Configuration:** 2048×2048 matrices, RTX 4070 Laptop GPU

| Kernel Implementation | Performance (GFLOPS) | Relative to cuBLAS |
|-----------------------|---------------------|-------------------|
| cuBLAS (Reference)    | 12,994.6           | 100.0%           |
| **Warp Tiled**        | **11,502.1**       | **88.5%**        |
| Vectorized            | 9,986.8            | 76.8%            |
| 2D Tiled              | 5,565.6            | 42.8%            |
| Shared Memory         | 1,898.3            | 14.6%            |
| Coalesced             | 1,461.1            | 11.2%            |
| Naive                 | 1,047.2            | 8.1%             |

**Achievement**: Custom Warp Tiled kernel achieves 88.5% of cuBLAS performance.

### Sparse Matrix Multiplication Results

**Small Matrix (500×500, 4,955 non-zeros):**

| Implementation      | Time (ms) | Performance (GFLOPS) | Speedup vs cuSPARSE |
|--------------------|-----------|---------------------|-------------------|
| Basic              | 0.013     | 0.8                | 3.0x              |
| Warp-reduction     | 0.013     | 0.8                | 3.0x              |
| Prefetch           | 0.013     | 0.8                | 3.0x              |
| Adaptive           | 0.013     | 0.8                | 3.0x              |
| cuSPARSE           | 0.040     | 0.251              | 1.0x (baseline)   |

### Conjugate Gradient Solver Results

**Problem:** 1024×1024 symmetric positive definite matrix, tolerance: 1e-10

| Implementation         | Total Time (ms) | Iterations | Final Residual | Accuracy        |
|------------------------|----------------|------------|----------------|-----------------|
| **Custom CUDA**        | **4.38**       | 20         | 2.862e-11     | 4.69e-07       |
| cuBLAS + cuSPARSE      | 10.62          | 20         | 2.862e-11     | 4.69e-07       |

**Results**: Custom implementation is 2.4× faster while maintaining identical numerical accuracy.

## Project Structure

```
CUDA/
├── CMakeLists.txt              # CMake build configuration
├── README.md                   # This file
├── cgSolver.cu                 # CG solver implementation
├── cgSolver.cuh                # CG solver header
├── cgSolverMain.cu             # CG solver main program
├── cgSolverRes.txt             # CG solver benchmark results
├── denseMatMul.cu              # Dense matrix multiplication kernels
├── denseMatMul.cuh             # Dense matrix multiplication header
├── denseMatMulMain.cu          # Dense matrix main program
├── denseMatMulRes.txt          # Dense matrix benchmark results
├── sparseMatMul.cu             # Sparse matrix multiplication kernels
├── sparseMatMul.cuh            # Sparse matrix multiplication header
├── sparseMatMulMain.cu         # Sparse matrix main program
└── sparseMatMulRes.txt         # Sparse matrix benchmark results
```

## Usage Examples

### Running Individual Components

1. **Dense Matrix Multiplication:**
   ```bash
   ./bin/denseMatMul
   ```

2. **Sparse Matrix Operations:**
   ```bash
   ./bin/sparseMatMul
   ```

3. **Conjugate Gradient Solver:**
   ```bash
   ./bin/cgSolver
   ```

### Custom Configuration

Each component supports configuration through source code modification:

- **Matrix sizes**: Modify `N` or `problemSize` constants
- **Convergence criteria**: Adjust `tolerance` and `maxIterations`
- **Sparsity patterns**: Customize matrix generation functions

## Technical Details

### Optimization Techniques

**Dense Matrix Multiplication:**
- Memory coalescing for optimal global memory access
- Shared memory tiling to reduce global memory traffic  
- Register blocking and vectorized loads
- Warp-level optimizations for maximum throughput

**Sparse Matrix Operations:**
- CSR format for efficient sparse storage
- Warp-level reductions for irregular memory patterns
- Cache optimization for repeated vector access
- Adaptive algorithms based on sparsity patterns

**Conjugate Gradient Solver:**
- GPU-parallel dot products and vector operations
- Sparse matrix-vector multiplication integration (custom vs cuSPARSE)
- Memory-efficient temporary storage management
- Numerical stability optimizations

### Hardware Requirements

- **Minimum**: NVIDIA GPU with Compute Capability 7.0 (Volta architecture)
- **Recommended**: RTX 30/40 series or A100/H100 for optimal performance
- **Memory**: 4GB+ GPU memory for large problem sizes

## License

This project is provided for educational and research purposes. Please ensure compliance with CUDA SDK license terms when using NVIDIA libraries.

## Acknowledgments

- Benchmarks and documentation generated using Claude AI assistant
- NVIDIA CUDA Toolkit and libraries (cuBLAS, cuSPARSE)
- Optimization techniques based on CUDA programming best practices
- Comparative analysis includes both custom implementations and NVIDIA's optimized libraries

---

*For questions or contributions, please refer to the source code comments for detailed implementation explanations.*
