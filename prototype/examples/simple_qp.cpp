#include "hybrid_active_nasoq.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

/**
 * This example solves a simple QP problem:
 * 
 * minimize    0.5*x'*H*x + q'*x
 * subject to  Ax = b
 *             Gx <= h
 * 
 * where:
 * H = [2 0; 0 2]
 * q = [-2; -2]
 * no equality constraints
 * G = [1 0; 0 1; -1 0; 0 -1]
 * h = [1; 1; 0; 0]
 * 
 * The optimal solution is x = [1; 1]
 */

int main() {
    using namespace nasoq;
    
    std::cout << "Simple QP Example using Hybrid Active-Projection Method\n";
    std::cout << "--------------------------------------------------------\n\n";
    
    // Define problem data
    
    // Hessian matrix (2x2 diagonal: [2 0; 0 2])
    int H_size = 2;
    int H_nnz = 2;
    
    int* Hp = new int[H_size + 1]{0, 1, 2};
    int* Hi = new int[H_nnz]{0, 1};
    double* Hx = new double[H_nnz]{2.0, 2.0};
    
    // Linear term (-2, -2)
    double* q = new double[H_size]{-2.0, -2.0};
    
    // No equality constraints
    int A_size1 = 0;
    int A_size2 = H_size;
    int* Ap = nullptr;
    int* Ai = nullptr;
    double* Ax = nullptr;
    double* a = nullptr;
    
    // Inequality constraints (box constraints: 0 <= x <= 1)
    int B_size1 = 4;
    int B_size2 = H_size;
    int B_nnz = 4;
    
    int* Bp = new int[B_size2 + 1]{0, 2, 4};
    int* Bi = new int[B_nnz]{0, 2, 1, 3};
    double* Bx = new double[B_nnz]{1.0, -1.0, 1.0, -1.0};
    double* b = new double[B_size1]{1.0, 0.0, 1.0, 0.0};
    
    // Create the solver
    HybridActiveNasoq solver(H_size, Hp, Hi, Hx, q, 
                          A_size1, A_size2, Ap, Ai, Ax, a,
                          B_size1, B_size2, Bp, Bi, Bx, b);
    
    // Set solver parameters
    solver.set_parameters(2, 1e-6, 100);
    solver.use_matrix_free_kkt(true);
    solver.enable_constraint_relaxation(true);
    solver.set_relaxation_parameters(0.1, 0.7, 0.001);
    
    // Solve the problem and measure time
    auto start_time = std::chrono::high_resolution_clock::now();
    
    int status = solver.solve();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Get and print results
    double* x = solver.get_primal_solution();
    
    std::cout << "Solver status: " << status << std::endl;
    std::cout << "Iterations: " << solver.get_iterations() << std::endl;
    std::cout << "Solve time: " << duration.count() / 1000.0 << " ms" << std::endl;
    std::cout << "Objective value: " << solver.compute_objective() << std::endl;
    std::cout << "Solution: [" << x[0] << ", " << x[1] << "]" << std::endl;
    
    // Active set
    const auto& active_set = solver.get_active_set();
    std::cout << "Active constraints: ";
    for (int idx : active_set) {
        std::cout << idx << " ";
    }
    std::cout << std::endl;
    
    // Clean up
    delete[] Hp;
    delete[] Hi;
    delete[] Hx;
    delete[] q;
    delete[] Bp;
    delete[] Bi;
    delete[] Bx;
    delete[] b;
    
    return 0;
} 