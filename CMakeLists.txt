cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(CUDA_LinearAlgebra_Toolkit LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDAToolkit REQUIRED)

# CUDA architecture configuration
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

# Compiler flags for optimization
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Add include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Create dense matrix multiplication executable
add_executable(denseMatMul
    denseMatMulMain.cu
    denseMatMul.cu
)

# Link cuBLAS library for dense operations
target_link_libraries(denseMatMul
    CUDA::cublas
    CUDA::cudart
)

# Create sparse matrix multiplication executable
add_executable(sparseMatMul
    sparseMatMulMain.cu
    sparseMatMul.cu
)

# Link cuSPARSE library for sparse operations
target_link_libraries(sparseMatMul
    CUDA::cusparse
    CUDA::cudart
)

# Create conjugate gradient solver executable
add_executable(cgSolver
    cgSolverMain.cu
    cgSolver.cu
    sparseMatMul.cu
    denseMatMul.cu
)

# Link required CUDA libraries for CG solver
target_link_libraries(cgSolver
    CUDA::cusparse
    CUDA::cublas
    CUDA::cudart
)

# Set output directory for executables
set_target_properties(denseMatMul sparseMatMul cgSolver
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Add custom target to run all benchmarks
add_custom_target(run_benchmarks
    COMMAND ${CMAKE_BINARY_DIR}/bin/denseMatMul > denseMatMulRes.txt
    COMMAND ${CMAKE_BINARY_DIR}/bin/sparseMatMul > sparseMatMulRes.txt
    COMMAND ${CMAKE_BINARY_DIR}/bin/cgSolver > cgSolverRes.txt
    DEPENDS denseMatMul sparseMatMul cgSolver
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running all benchmarks and generating result files"
)

# Installation rules
install(TARGETS denseMatMul sparseMatMul cgSolver
    RUNTIME DESTINATION bin
)

# Print configuration summary
message(STATUS "CUDA Linear Algebra Toolkit Configuration:")
message(STATUS "  CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Standard: ${CMAKE_CUDA_STANDARD}")
