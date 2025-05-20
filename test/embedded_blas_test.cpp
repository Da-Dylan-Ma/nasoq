#include "nasoq/embedded/embedded_blas.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

// Helper function to print a vector
void print_vector(const std::string& name, const std::vector<double>& vec) {
    std::cout << name << " = [";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << std::setw(8) << std::fixed << std::setprecision(4) << vec[i];
        if (i < vec.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

// Helper function to print a matrix
void print_matrix(const std::string& name, const std::vector<double>& mat, int rows, int cols) {
    std::cout << name << " = [" << std::endl;
    for (int i = 0; i < rows; ++i) {
        std::cout << "  ";
        for (int j = 0; j < cols; ++j) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(4) << mat[i + j * rows];
            if (j < cols - 1) std::cout << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << "]" << std::endl;
}

// Test dgemv implementation
void test_dgemv() {
    std::cout << "\n===== Testing dgemv =====" << std::endl;
    
    // Matrix A (column-major order)
    std::vector<double> A = {
        1.0, 4.0, 7.0,  // First column
        2.0, 5.0, 8.0,  // Second column
        3.0, 6.0, 9.0   // Third column
    };
    
    // Vector x
    std::vector<double> x = {1.0, 2.0, 3.0};
    
    // Vector y for output
    std::vector<double> y = {0.0, 0.0, 0.0};
    
    int m = 3; // Rows of A
    int n = 3; // Columns of A
    char trans = 'N'; // No transpose
    double alpha = 1.0;
    double beta = 0.0;
    int lda = 3; // Leading dimension of A
    int incx = 1; // Stride of x
    int incy = 1; // Stride of y
    
    // Print inputs
    print_matrix("Matrix A", A, m, n);
    print_vector("Vector x", x);
    print_vector("Initial y", y);
    
    // Call dgemv
    nasoq::embedded::dgemv(&trans, &m, &n, &alpha, A.data(), &lda, 
                         x.data(), &incx, &beta, y.data(), &incy);
    
    // Print result and expected
    print_vector("Result y", y);
    std::cout << "Expected y = [14.0000, 32.0000, 50.0000]" << std::endl;
    
    // Test transpose operation
    std::fill(y.begin(), y.end(), 0.0);
    trans = 'T';
    
    std::cout << "\nTesting transpose operation:" << std::endl;
    print_vector("Initial y", y);
    
    nasoq::embedded::dgemv(&trans, &m, &n, &alpha, A.data(), &lda, 
                         x.data(), &incx, &beta, y.data(), &incy);
    
    print_vector("Result y (A'*x)", y);
    std::cout << "Expected y = [30.0000, 36.0000, 42.0000]" << std::endl;
}

// Test blocked_2by2_mult implementation
void test_blocked_2by2_mult() {
    std::cout << "\n===== Testing blocked_2by2_mult =====" << std::endl;
    
    // Test case 1: 1x1 blocks
    {
        // Diagonal D (diagonal elements and subdiagonal elements)
        std::vector<double> D = {2.0, 3.0, 0.0, 0.0};
        
        // Source matrix (2 rows x 2 columns, column-major)
        std::vector<double> src = {1.0, 4.0, 2.0, 5.0};
        
        // Destination matrix
        std::vector<double> dst(4, 0.0);
        
        int n = 2; // Number of diagonal blocks
        int m = 2; // Number of columns in src/dst
        int lda = 1; // Leading dimension of src
        int lda_d = 2; // Stride for subdiagonal elements
        
        // Print inputs
        print_vector("Diagonal D", {D[0], D[1]});
        print_matrix("Source matrix", src, n, m);
        
        // Call blocked_2by2_mult
        nasoq::embedded::blocked_2by2_mult(n, m, D.data(), src.data(), dst.data(), lda, lda_d);
        
        // Print result
        print_matrix("Result matrix", dst, n, m);
        std::cout << "Expected matrix = [" << std::endl;
        std::cout << "  2.0000, 4.0000" << std::endl;
        std::cout << "  12.0000, 15.0000" << std::endl;
        std::cout << "]" << std::endl;
    }
    
    // Test case 2: with 2x2 block
    {
        // Block diagonal matrix (2x2 block)
        // [ 2.0  1.0 ]
        // [ 1.0  3.0 ]
        std::vector<double> D = {2.0, 3.0, 1.0};
        
        // Source matrix (2 rows x 2 columns, column-major)
        std::vector<double> src = {1.0, 4.0, 2.0, 5.0};
        
        // Destination matrix
        std::vector<double> dst(4, 0.0);
        
        int n = 2; // Size of D
        int m = 2; // Number of columns in src/dst
        int lda = 1; // Leading dimension of src
        int lda_d = 2; // Stride for subdiagonal elements
        
        // Print inputs
        std::cout << "\nTest with 2x2 block:" << std::endl;
        std::cout << "Block diagonal D = [ 2.0, 1.0; 1.0, 3.0 ]" << std::endl;
        print_matrix("Source matrix", src, n, m);
        
        // Call blocked_2by2_mult
        nasoq::embedded::blocked_2by2_mult(n, m, D.data(), src.data(), dst.data(), lda, lda_d);
        
        // Print result
        print_matrix("Result matrix", dst, n, m);
        std::cout << "Expected matrix = [" << std::endl;
        std::cout << "  6.0000, 9.0000" << std::endl;
        std::cout << "  7.0000, 19.0000" << std::endl;
        std::cout << "]" << std::endl;
    }
}

int main() {
    std::cout << "Testing embedded BLAS implementations" << std::endl;
    
    test_dgemv();
    test_blocked_2by2_mult();
    
    std::cout << "\nAll tests completed!" << std::endl;
    return 0;
} 