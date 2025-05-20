#include <gtest/gtest.h>
#include "hybrid_active_nasoq.h"
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

using namespace nasoq;

// Helper function to create a random QP problem
void create_random_qp(int n, int m, 
                     int** Hp, int** Hi, double** Hx, double** q,
                     int** Bp, int** Bi, double** Bx, double** b) {
    // Set up random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.1, 1.0);
    
    // Create a positive definite Hessian matrix (diagonal for simplicity)
    *Hp = new int[n + 1];
    *Hi = new int[n];
    *Hx = new double[n];
    
    for (int i = 0; i <= n; i++) {
        (*Hp)[i] = i;
    }
    
    for (int i = 0; i < n; i++) {
        (*Hi)[i] = i;
        (*Hx)[i] = dis(gen) + n; // Ensure positive definite
    }
    
    // Create a linear term
    *q = new double[n];
    for (int i = 0; i < n; i++) {
        (*q)[i] = dis(gen) - 0.5;
    }
    
    // Create inequality constraints (one per variable, plus some random ones)
    int nnz = n + m;
    *Bp = new int[n + 1];
    *Bi = new int[nnz];
    *Bx = new int[nnz];
    
    // Set up column pointers
    for (int i = 0; i <= n; i++) {
        (*Bp)[i] = i * (1 + m/n);
    }
    (*Bp)[n] = nnz;
    
    // Set up row indices and values
    for (int i = 0; i < nnz; i++) {
        (*Bi)[i] = i % m;
        (*Bx)[i] = dis(gen) - 0.5;
    }
    
    // Create RHS
    *b = new double[m];
    for (int i = 0; i < m; i++) {
        (*b)[i] = dis(gen);
    }
}

// Test that different batch sizes all converge to the same solution
TEST(BatchProcessingTest, DifferentBatchSizesSameSolution) {
    // Problem dimensions
    const int n = 20;  // Variables
    const int m = 50;  // Constraints
    
    // Create random QP problem
    int *Hp, *Hi, *Bp, *Bi;
    double *Hx, *q, *Bx, *b;
    create_random_qp(n, m, &Hp, &Hi, &Hx, &q, &Bp, &Bi, &Bx, &b);
    
    // Reference solution with batch size 1 (traditional active-set)
    HybridActiveNasoq reference_solver(n, Hp, Hi, Hx, q, 
                                     0, 0, nullptr, nullptr, nullptr, nullptr,
                                     m, n, Bp, Bi, Bx, b);
    reference_solver.set_parameters(1, 1e-6, 1000);
    reference_solver.use_matrix_free_kkt(false);
    reference_solver.enable_constraint_relaxation(false);
    
    int reference_status = reference_solver.solve();
    ASSERT_EQ(reference_status, CONVERGED);
    
    double* reference_x = reference_solver.get_primal_solution();
    double reference_obj = reference_solver.compute_objective();
    int reference_iterations = reference_solver.get_iterations();
    
    // Test different batch sizes
    const std::vector<int> batch_sizes = {2, 5, 10, 20};
    std::vector<int> iterations;
    
    for (int batch_size : batch_sizes) {
        HybridActiveNasoq solver(n, Hp, Hi, Hx, q, 
                               0, 0, nullptr, nullptr, nullptr, nullptr,
                               m, n, Bp, Bi, Bx, b);
        solver.set_parameters(batch_size, 1e-6, 1000);
        solver.use_matrix_free_kkt(false);
        solver.enable_constraint_relaxation(false);
        
        auto start = std::chrono::high_resolution_clock::now();
        int status = solver.solve();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        ASSERT_EQ(status, CONVERGED);
        
        double* x = solver.get_primal_solution();
        double obj = solver.compute_objective();
        iterations.push_back(solver.get_iterations());
        
        // Check that objective values are very close
        EXPECT_NEAR(obj, reference_obj, 1e-4);
        
        // Check that solution vectors are close
        for (int i = 0; i < n; i++) {
            EXPECT_NEAR(x[i], reference_x[i], 1e-4);
        }
        
        std::cout << "Batch size " << batch_size 
                  << ", iterations: " << solver.get_iterations()
                  << ", time: " << duration.count() << " ms" << std::endl;
    }
    
    // Verify that larger batch sizes generally require fewer iterations
    // (but not necessarily monotonically)
    int min_batched_iterations = *std::min_element(iterations.begin(), iterations.end());
    EXPECT_LT(min_batched_iterations, reference_iterations);
    
    // Clean up
    delete[] Hp;
    delete[] Hi;
    delete[] Hx;
    delete[] q;
    delete[] Bp;
    delete[] Bi;
    delete[] Bx;
    delete[] b;
}

// Test with constraint relaxation enabled
TEST(BatchProcessingTest, ConstraintRelaxation) {
    // Problem dimensions
    const int n = 20;  // Variables
    const int m = 50;  // Constraints
    
    // Create random QP problem
    int *Hp, *Hi, *Bp, *Bi;
    double *Hx, *q, *Bx, *b;
    create_random_qp(n, m, &Hp, &Hi, &Hx, &q, &Bp, &Bi, &Bx, &b);
    
    // Solve without relaxation
    HybridActiveNasoq no_relax_solver(n, Hp, Hi, Hx, q, 
                                    0, 0, nullptr, nullptr, nullptr, nullptr,
                                    m, n, Bp, Bi, Bx, b);
    no_relax_solver.set_parameters(5, 1e-6, 1000);
    no_relax_solver.use_matrix_free_kkt(false);
    no_relax_solver.enable_constraint_relaxation(false);
    
    auto start1 = std::chrono::high_resolution_clock::now();
    int status1 = no_relax_solver.solve();
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
    
    ASSERT_EQ(status1, CONVERGED);
    double obj1 = no_relax_solver.compute_objective();
    int iter1 = no_relax_solver.get_iterations();
    
    // Solve with relaxation
    HybridActiveNasoq relax_solver(n, Hp, Hi, Hx, q, 
                                 0, 0, nullptr, nullptr, nullptr, nullptr,
                                 m, n, Bp, Bi, Bx, b);
    relax_solver.set_parameters(5, 1e-6, 1000);
    relax_solver.use_matrix_free_kkt(false);
    relax_solver.enable_constraint_relaxation(true);
    relax_solver.set_relaxation_parameters(0.1, 0.7, 0.001);
    
    auto start2 = std::chrono::high_resolution_clock::now();
    int status2 = relax_solver.solve();
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
    
    ASSERT_EQ(status2, CONVERGED);
    double obj2 = relax_solver.compute_objective();
    int iter2 = relax_solver.get_iterations();
    
    // Check that both approaches converge to the same solution
    EXPECT_NEAR(obj1, obj2, 1e-4);
    
    std::cout << "Without relaxation: iterations=" << iter1 
              << ", time=" << duration1.count() << "ms" << std::endl;
    std::cout << "With relaxation: iterations=" << iter2 
              << ", time=" << duration2.count() << "ms" << std::endl;
    
    // Clean up
    delete[] Hp;
    delete[] Hi;
    delete[] Hx;
    delete[] q;
    delete[] Bp;
    delete[] Bi;
    delete[] Bx;
    delete[] b;
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 