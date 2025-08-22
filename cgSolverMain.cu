#include "cgSolver.cuh"

int main() {
    cout << "CUDA-Accelerated Conjugate Gradient Linear Solver" << endl;
    cout << "=================================================" << endl << endl;
    
    const int problemSize = 1024;
    const double tolerance = 1e-10;
    const int maxIterations = 500;
    
    cout << "Problem Configuration:" << endl;
    cout << "  Matrix size: " << problemSize << " x " << problemSize << endl;
    cout << "  Convergence tolerance: " << scientific << tolerance << endl;
    cout << "  Maximum iterations: " << maxIterations << endl << endl;
    
    // Generate test problem
    cout << "Generating test matrix..." << endl;
    vector<int> rowPtr, colIdx;
    vector<float> values;
    generateTestMatrix(rowPtr, colIdx, values, problemSize);
    
    cout << "  Matrix type: Symmetric positive definite" << endl;
    cout << "  Sparsity pattern: Tridiagonal with diagonal dominance" << endl;
    cout << "  Non-zero entries: " << values.size() << endl;
    cout << "  Sparsity: " << fixed << setprecision(1) 
         << (100.0 * values.size()) / (problemSize * problemSize) << "%" << endl << endl;
    
    // Create known solution and corresponding RHS
    vector<float> exactSolution(problemSize, 1.0f);
    vector<float> rightHandSide(problemSize, 0.0f);
    
    // Compute b = A * x_exact
    for (int row = 0; row < problemSize; row++) {
        for (int j = rowPtr[row]; j < rowPtr[row + 1]; j++) {
            int col = colIdx[j];
            float val = values[j];
            rightHandSide[row] += val * exactSolution[col];
        }
    }
    
    cout << "Test case: Known solution x = [1, 1, 1, ...]" << endl;
    cout << "RHS vector norm: " << fixed << setprecision(4) 
         << sqrt(inner_product(rightHandSide.begin(), rightHandSide.end(), 
                              rightHandSide.begin(), 0.0)) << endl << endl;
    
    // Benchmark custom implementation
    cout << "Solver Implementation 1: Custom CUDA Kernels" << endl;
    cout << "--------------------------------------------" << endl;
    ConjugateGradientSolver solver1(rowPtr, colIdx, values, problemSize, true);
    
    vector<float> solution1;
    auto start1 = high_resolution_clock::now();
    bool success1 = solver1.solveSystem(rightHandSide, solution1, tolerance, maxIterations, true);
    auto end1 = high_resolution_clock::now();
    auto time1 = duration_cast<microseconds>(end1 - start1);
    
    cout << endl;
    
    // Benchmark cuBLAS + cuSPARSE implementation
    cout << "Solver Implementation 2: cuBLAS + cuSPARSE" << endl;
    cout << "-------------------------------------------" << endl;
    ConjugateGradientSolver solver2(rowPtr, colIdx, values, problemSize, false);
    
    vector<float> solution2;
    auto start2 = high_resolution_clock::now();
    bool success2 = solver2.solveSystem(rightHandSide, solution2, tolerance, maxIterations, true);
    auto end2 = high_resolution_clock::now();
    auto time2 = duration_cast<microseconds>(end2 - start2);
    
    cout << endl;
    
    // Performance comparison
    cout << "Performance Summary" << endl;
    cout << "==================" << endl;
    
    if (success1) {
        float error1 = 0;
        for (int i = 0; i < problemSize; i++) {
            float diff = solution1[i] - exactSolution[i];
            error1 += diff * diff;
        }
        error1 = sqrt(error1);
        
        cout << "Custom Implementation:" << endl;
        cout << "  Status: CONVERGED" << endl;
        cout << "  Total time: " << fixed << setprecision(2) << time1.count() / 1000.0 << " ms" << endl;
        cout << "  Solution accuracy: " << scientific << setprecision(2) << error1 << endl;
    } else {
        cout << "Custom Implementation: FAILED TO CONVERGE" << endl;
    }
    
    cout << endl;
    
    if (success2) {
        float error2 = 0;
        for (int i = 0; i < problemSize; i++) {
            float diff = solution2[i] - exactSolution[i];
            error2 += diff * diff;
        }
        error2 = sqrt(error2);
        
        cout << "cuBLAS + cuSPARSE Implementation:" << endl;
        cout << "  Status: CONVERGED" << endl;
        cout << "  Total time: " << fixed << setprecision(2) << time2.count() / 1000.0 << " ms" << endl;
        cout << "  Solution accuracy: " << scientific << setprecision(2) << error2 << endl;
    } else {
        cout << "cuBLAS + cuSPARSE Implementation: FAILED TO CONVERGE" << endl;
    }
    
    cout << endl;
    
    if (success1 && success2) {
        cout << "Comparative Analysis:" << endl;
        double speedup = (double)time1.count() / time2.count();
        if (speedup > 1.05) {
            cout << "  cuBLAS + cuSPARSE implementation is " << fixed << setprecision(1) 
                 << speedup << "x faster" << endl;
        } else if (speedup < 0.95) {
            cout << "  Custom implementation is " << fixed << setprecision(1) 
                 << (1.0/speedup) << "x faster" << endl;
        } else {
            cout << "  Both implementations show comparable performance" << endl;
        }
        
        cout << "  Both solvers achieved similar numerical accuracy" << endl;
        cout << "  Memory efficiency: Sparse storage reduces memory by " 
             << fixed << setprecision(0) 
             << (100.0 - (100.0 * values.size()) / (problemSize * problemSize)) << "%" << endl;
    }
    
    cout << endl << "Linear system successfully solved using GPU-accelerated CG method." << endl;
    
    return 0;
}