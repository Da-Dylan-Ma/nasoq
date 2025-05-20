#include <iostream>
#include <cassert>
#include <iomanip>
#include <map>
#include <string>
#include <vector>
#include "nasoq/embedded/embedded_blas.h"

// Helper function to check if two values are approximately equal
bool approx_equal(double a, double b, double epsilon = 1e-10) {
    return std::abs(a - b) < epsilon;
}

// Custom test framework to continue on failures
struct TestFunction {
    std::string name;
    int passed = 0;
    int failed = 0;
    std::vector<std::string> failures;
    
    // Add a constructor that takes a name parameter
    TestFunction() {}
    TestFunction(const std::string& test_name) : name(test_name) {}
};

std::map<std::string, TestFunction> test_functions;
std::string current_test_function = "";

// Start a new test function
void begin_test(const std::string& name) {
    current_test_function = name;
    std::cout << "\n===== Testing " << name << " =====" << std::endl;
    
    // Initialize if not already present
    if (test_functions.find(name) == test_functions.end()) {
        test_functions[name] = TestFunction{name};
    }
}

// Custom assertion that logs failure but continues execution
void test_assert(bool condition, const std::string& message) {
    TestFunction& func = test_functions[current_test_function];
    
    if (condition) {
        func.passed++;
    } else {
        func.failed++;
        func.failures.push_back(message);
        std::cout << "ASSERTION FAILED: " << message << std::endl;
    }
}

// Helper function to print a matrix
void print_matrix(const double* matrix, int rows, int cols, int ld) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) << matrix[i + j*ld] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    std::cout << "Starting embedded BLAS functions test..." << std::endl;
    
    //====================================================================
    // Test dscal
    //====================================================================
    begin_test("dscal");
    const int n = 5;
    const double alpha = 2.0;
    double x[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    int incx = 1;
    
    std::cout << "Original x values:" << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << x[i] << " ";
    }
    std::cout << std::endl;
    
    // Call embedded dscal
    nasoq::embedded::dscal(&n, &alpha, x, &incx);
    
    std::cout << "After dscal (should be multiplied by 2.0):" << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << x[i] << " ";
    }
    std::cout << std::endl;
    
    // Verify results
    test_assert(approx_equal(x[0], 2.0), "x[0] should be 2.0");
    test_assert(approx_equal(x[1], 4.0), "x[1] should be 4.0");
    test_assert(approx_equal(x[2], 6.0), "x[2] should be 6.0");
    test_assert(approx_equal(x[3], 8.0), "x[3] should be 8.0");
    test_assert(approx_equal(x[4], 10.0), "x[4] should be 10.0");
    
    //====================================================================
    // Test dsyr
    //====================================================================
    begin_test("dsyr");
    const int m = 3;
    
    // Initialize a 3x3 identity matrix in column-major order (standard for BLAS)
    double a[9] = {1.0, 0.0, 0.0,  // First column
                   0.0, 1.0, 0.0,  // Second column
                   0.0, 0.0, 1.0}; // Third column
    
    // Initialize vector x = [1, 2, 3]
    double y[3] = {1.0, 2.0, 3.0};
    
    // Use alpha = 2.0 to match both NASOQ usage and previous test
    const double beta = 2.0;
    
    // Use lower triangular update mode
    const char uplo = 'L';
    
    int incy = 1;
    int lda = 3;
    
    std::cout << "Original matrix a:" << std::endl;
    print_matrix(a, m, m, lda);
    
    // Call embedded dsyr
    nasoq::embedded::dsyr(&uplo, &m, &beta, y, &incy, a, &lda);
    
    std::cout << "After dsyr (rank-1 update with y and beta=2.0):" << std::endl;
    print_matrix(a, m, m, lda);
    
    // For dsyr update with alpha=2.0, x=[1,2,3], and starting with identity matrix:
    // The L part of result should be:
    // [ 1 + 2*1*1,      0,      0 ]   [ 3,  0,  0 ]
    // [ 2*1*2,    1 + 2*2*2,    0 ] = [ 4,  9,  0 ]
    // [ 2*1*3,      2*2*3,  1 + 2*3*3 ] [ 6, 12, 19 ]
    
    // Verify results with detailed comments to ensure correctness
    test_assert(approx_equal(a[0], 3.0), "a[0,0] = 1 + 2*1*1 should be 3.0");
    test_assert(approx_equal(a[3], 4.0), "a[1,0] = 2*1*2 should be 4.0");
    test_assert(approx_equal(a[4], 9.0), "a[1,1] = 1 + 2*2*2 should be 9.0");
    test_assert(approx_equal(a[6], 6.0), "a[2,0] = 2*1*3 should be 6.0");
    test_assert(approx_equal(a[7], 12.0), "a[2,1] = 2*2*3 should be 12.0");
    test_assert(approx_equal(a[8], 19.0), "a[2,2] = 1 + 2*3*3 should be 19.0");
    
    //====================================================================
    // Test dcopy
    //====================================================================
    begin_test("dcopy");
    const int n_copy = 4;
    double src[4] = {1.0, 2.0, 3.0, 4.0};
    double dst[4] = {0.0, 0.0, 0.0, 0.0};
    int inc_src = 1;
    int inc_dst = 1;
    
    std::cout << "Source array:" << std::endl;
    for (int i = 0; i < n_copy; i++) {
        std::cout << src[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Destination array before copy:" << std::endl;
    for (int i = 0; i < n_copy; i++) {
        std::cout << dst[i] << " ";
    }
    std::cout << std::endl;
    
    // Call embedded dcopy
    nasoq::embedded::dcopy(&n_copy, src, &inc_src, dst, &inc_dst);
    
    std::cout << "Destination array after copy:" << std::endl;
    for (int i = 0; i < n_copy; i++) {
        std::cout << dst[i] << " ";
    }
    std::cout << std::endl;
    
    // Verify results
    for (int i = 0; i < n_copy; i++) {
        test_assert(approx_equal(dst[i], src[i]), "dst[" + std::to_string(i) + "] should be src[" + std::to_string(i) + "]");
    }
    
    //====================================================================
    // Test blocked_2by2_solver
    //====================================================================
    begin_test("blocked_2by2_solver");
    
    // 1x1 and 2x2 blocks test case
    int n_blk = 3;
    double D[6] = {2.0, 3.0, 4.0, 0.0, 1.0, 0.0}; // D[0]=2.0, D[1]=3.0 with off-diagonal D[4]=1.0, D[2]=4.0
    double rhs[6] = {4.0, 11.0, 12.0, 6.0, 22.0, 24.0}; // 3x2 right-hand sides
    int n_rhs = 2;
    int lda_rhs = 3;
    int lda_d = 3;
    
    std::cout << "Diagonal matrix D:" << std::endl;
    std::cout << "Diag:   " << D[0] << " " << D[1] << " " << D[2] << std::endl;
    std::cout << "Off-diag: " << D[3] << " " << D[4] << " " << D[5] << std::endl;
    
    std::cout << "Right-hand sides before solve:" << std::endl;
    print_matrix(rhs, n_blk, n_rhs, lda_rhs);
    
    // Call embedded blocked_2by2_solver
    nasoq::embedded::blocked_2by2_solver(n_blk, D, rhs, n_rhs, lda_rhs, lda_d);
    
    std::cout << "Solution after solve:" << std::endl;
    print_matrix(rhs, n_blk, n_rhs, lda_rhs);
    
    // Verify results - first test with original D and solution to confirm correctness
    // For first block (2x2): [2.0 1.0; 1.0 3.0] * [rhs[0]; rhs[1]] = [4.0; 11.0]
    double sol_check1 = D[0] * rhs[0] + D[4] * rhs[1];
    double sol_check2 = D[4] * rhs[0] + D[1] * rhs[1];
    test_assert(approx_equal(sol_check1, 4.0), "First block solution should be 4.0");
    test_assert(approx_equal(sol_check2, 11.0), "First block solution should be 11.0");
    
    // For 1x1 block: 4.0 * rhs[2] = 12.0
    test_assert(approx_equal(D[2] * rhs[2], 12.0), "1x1 block solution should be 12.0");
    
    //====================================================================
    // Test dgemv
    //====================================================================
    begin_test("dgemv");
    
    // Test case 1: Non-transpose operation
    {
        const int rows = 3;
        const int cols = 2;
        double mat_A[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // Column-major: [1 4; 2 5; 3 6]
        double vec_x[2] = {2.0, 3.0};
        double vec_y[3] = {1.0, 1.0, 1.0};
        double alpha_val = 1.0;
        double beta_val = 1.0;
        char trans_val = 'N';
        int ld_a = 3;
        int inc_x = 1;
        int inc_y = 1;
        
        std::cout << "Matrix A:" << std::endl;
        print_matrix(mat_A, rows, cols, ld_a);
        
        std::cout << "Vector x: ";
        for (int i = 0; i < cols; i++) std::cout << vec_x[i] << " ";
        std::cout << std::endl;
        
        std::cout << "Vector y before dgemv: ";
        for (int i = 0; i < rows; i++) std::cout << vec_y[i] << " ";
        std::cout << std::endl;
        
        // Call embedded dgemv
        nasoq::embedded::dgemv(&trans_val, &rows, &cols, &alpha_val, mat_A, &ld_a, 
                             vec_x, &inc_x, &beta_val, vec_y, &inc_y);
        
        std::cout << "Vector y after dgemv: ";
        for (int i = 0; i < rows; i++) std::cout << vec_y[i] << " ";
        std::cout << std::endl;
        
        // Expected result: y = alpha*A*x + beta*y
        // [1 4]   [2]   [1]   [1+2+12]   [15]
        // [2 5] * [3] + [1] = [1+4+15] = [20]
        // [3 6]         [1]   [1+6+18]   [25]
        
        // Verify results
        test_assert(approx_equal(vec_y[0], 15.0), "vec_y[0] should be 15.0");
        test_assert(approx_equal(vec_y[1], 20.0), "vec_y[1] should be 20.0");
        test_assert(approx_equal(vec_y[2], 25.0), "vec_y[2] should be 25.0");
    }
    
    // Test case 2: Transpose operation
    {
        const int rows = 3;
        const int cols = 2;
        double mat_A[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // Column-major: [1 4; 2 5; 3 6]
        double vec_x[3] = {2.0, 3.0, 4.0};
        double vec_y[2] = {1.0, 1.0};
        double alpha_val = 2.0;
        double beta_val = 0.5;
        char trans_val = 'T';
        int ld_a = 3;
        int inc_x = 1;
        int inc_y = 1;
        
        std::cout << "\nMatrix A:" << std::endl;
        print_matrix(mat_A, rows, cols, ld_a);
        
        std::cout << "Vector x: ";
        for (int i = 0; i < rows; i++) std::cout << vec_x[i] << " ";
        std::cout << std::endl;
        
        std::cout << "Vector y before dgemv: ";
        for (int i = 0; i < cols; i++) std::cout << vec_y[i] << " ";
        std::cout << std::endl;
        
        // Call embedded dgemv
        nasoq::embedded::dgemv(&trans_val, &rows, &cols, &alpha_val, mat_A, &ld_a, 
                             vec_x, &inc_x, &beta_val, vec_y, &inc_y);
        
        std::cout << "Vector y after dgemv: ";
        for (int i = 0; i < cols; i++) std::cout << vec_y[i] << " ";
        std::cout << std::endl;
        
        // Expected result: y = alpha*A'*x + beta*y
        // [1 2 3]   [2]   [0.5]   [2*(2+6+12)+0.5]   [40.5]
        // [4 5 6] * [3] + [0.5] = [2*(8+15+24)+0.5] = [94.5]
        //           [4]
        
        // Verify results
        test_assert(approx_equal(vec_y[0], 40.5), "vec_y[0] should be 40.5");
        test_assert(approx_equal(vec_y[1], 94.5), "vec_y[1] should be 94.5");
    }
    
    //====================================================================
    // Test blocked_2by2_mult
    //====================================================================
    begin_test("blocked_2by2_mult");
    
    // Test case 1: 1x1 blocks
    {
        int n_blk_mult = 2;
        int m_blk_mult = 2;
        double D_mult[4] = {2.0, 3.0, 0.0, 0.0}; // Diagonal blocks (D[0]=2.0, D[1]=3.0, no 2x2 blocks)
        double src_mult[4] = {1.0, 4.0, 2.0, 5.0}; // Source matrix (column-major): [1 2; 4 5]
        double dst_mult[4] = {0.0, 0.0, 0.0, 0.0}; // Destination matrix
        int ld_src = 1; // Leading dimension of src
        int ld_d = 2; // Stride for subdiagonal elements
        
        std::cout << "Diagonal D: ";
        for (int i = 0; i < n_blk_mult; i++) std::cout << D_mult[i] << " ";
        std::cout << std::endl;
        
        std::cout << "Source matrix:" << std::endl;
        for (int j = 0; j < m_blk_mult; j++) {
            for (int i = 0; i < n_blk_mult; i++) {
                std::cout << src_mult[i*ld_src + j] << " ";
            }
            std::cout << std::endl;
        }
        
        // Call embedded blocked_2by2_mult
        nasoq::embedded::blocked_2by2_mult(n_blk_mult, m_blk_mult, D_mult, src_mult, dst_mult, ld_src, ld_d);
        
        std::cout << "Result matrix:" << std::endl;
        for (int j = 0; j < m_blk_mult; j++) {
            for (int i = 0; i < n_blk_mult; i++) {
                std::cout << dst_mult[i*m_blk_mult + j] << " ";
            }
            std::cout << std::endl;
        }
        
        // Expected result:
        // [2 0]   [1 2]   [2*1 2*2]   [2 4]
        // [0 3] * [4 5] = [3*4 3*5] = [12 15]
        
        // Verify results
        test_assert(approx_equal(dst_mult[0], 2.0), "dst_mult[0] should be 2.0");
        test_assert(approx_equal(dst_mult[1], 12.0), "dst_mult[1] should be 12.0");
        test_assert(approx_equal(dst_mult[2], 4.0), "dst_mult[2] should be 4.0");
        test_assert(approx_equal(dst_mult[3], 15.0), "dst_mult[3] should be 15.0");
    }
    
    // Test case 2: with 2x2 block
    {
        int n_blk_mult = 2;
        int m_blk_mult = 2;
        double D_mult[3] = {2.0, 3.0, 1.0}; // D[0]=2.0, D[1]=3.0, D[2]=subdiagonal=1.0
        double src_mult[4] = {1.0, 4.0, 2.0, 5.0}; // Source matrix (column-major): [1 2; 4 5]
        double dst_mult[4] = {0.0, 0.0, 0.0, 0.0}; // Destination matrix
        int ld_src = 1; // Leading dimension of src
        int ld_d = 2; // Stride for subdiagonal elements
        
        std::cout << "\nBlock diagonal D = [ 2.0, 1.0; 1.0, 3.0 ]" << std::endl;
        
        std::cout << "Source matrix:" << std::endl;
        for (int j = 0; j < m_blk_mult; j++) {
            for (int i = 0; i < n_blk_mult; i++) {
                std::cout << src_mult[i*ld_src + j] << " ";
            }
            std::cout << std::endl;
        }
        
        // Call embedded blocked_2by2_mult
        nasoq::embedded::blocked_2by2_mult(n_blk_mult, m_blk_mult, D_mult, src_mult, dst_mult, ld_src, ld_d);
        
        std::cout << "Result matrix:" << std::endl;
        for (int j = 0; j < m_blk_mult; j++) {
            for (int i = 0; i < n_blk_mult; i++) {
                std::cout << dst_mult[i*m_blk_mult + j] << " ";
            }
            std::cout << std::endl;
        }
        
        // Expected result:
        // [2 1]   [1 2]   [2*1+1*4 2*2+1*5]   [6 9]
        // [1 3] * [4 5] = [1*1+3*4 1*2+3*5] = [13 17]
        
        // Verify results
        test_assert(approx_equal(dst_mult[0], 6.0), "dst_mult[0] should be 6.0");
        test_assert(approx_equal(dst_mult[1], 13.0), "dst_mult[1] should be 13.0");
        test_assert(approx_equal(dst_mult[2], 9.0), "dst_mult[2] should be 9.0");
        test_assert(approx_equal(dst_mult[3], 17.0), "dst_mult[3] should be 17.0");
    }
    
    //====================================================================
    // Test dgemm
    //====================================================================
    begin_test("dgemm");
    
    // Test case 1: No transpose
    {
        const int m_gemm = 2; // Rows of C and A
        const int n_gemm = 3; // Columns of C and B
        const int k_gemm = 2; // Columns of A, rows of B
        
        // Matrix A (column-major): [1 3; 2 4]
        double a_gemm[4] = {1.0, 2.0, 3.0, 4.0};
        
        // Matrix B (column-major): [1 3 5; 2 4 6]
        double b_gemm[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        
        // Matrix C (column-major, will be overwritten): [1 4 7; 2 5 8]
        double c_gemm[6] = {1.0, 2.0, 4.0, 5.0, 7.0, 8.0};
        
        double alpha_gemm = 1.0;
        double beta_gemm = 0.0; // C will be completely overwritten
        
        char transa_gemm = 'N';
        char transb_gemm = 'N';
        
        int lda_gemm = 2;
        int ldb_gemm = 2;
        int ldc_gemm = 2;
        
        std::cout << "Matrix A:" << std::endl;
        print_matrix(a_gemm, m_gemm, k_gemm, lda_gemm);
        
        std::cout << "Matrix B:" << std::endl;
        print_matrix(b_gemm, k_gemm, n_gemm, ldb_gemm);
        
        std::cout << "Matrix C before dgemm:" << std::endl;
        print_matrix(c_gemm, m_gemm, n_gemm, ldc_gemm);
        
        // Call embedded dgemm
        nasoq::embedded::dgemm(&transa_gemm, &transb_gemm, &m_gemm, &n_gemm, &k_gemm, 
                             &alpha_gemm, a_gemm, &lda_gemm, b_gemm, &ldb_gemm, 
                             &beta_gemm, c_gemm, &ldc_gemm);
        
        std::cout << "Matrix C after dgemm:" << std::endl;
        print_matrix(c_gemm, m_gemm, n_gemm, ldc_gemm);
        
        // Expected result: C = A*B
        // [1 3]   [1 3 5]   [7  15  23]
        // [2 4] * [2 4 6] = [10 22  34]
        
        // Check results
        test_assert(approx_equal(c_gemm[0], 7.0), "c[0,0] should be 7.0");
        test_assert(approx_equal(c_gemm[1], 10.0), "c[1,0] should be 10.0");
        test_assert(approx_equal(c_gemm[2], 15.0), "c[0,1] should be 15.0");
        test_assert(approx_equal(c_gemm[3], 22.0), "c[1,1] should be 22.0");
        test_assert(approx_equal(c_gemm[4], 23.0), "c[0,2] should be 23.0");
        test_assert(approx_equal(c_gemm[5], 34.0), "c[1,2] should be 34.0");
    }
    
    // Test case 2: With transpose A
    {
        const int m_gemm = 2; // Rows of C and op(A)
        const int n_gemm = 2; // Columns of C and op(B)
        const int k_gemm = 3; // Columns of op(A), rows of op(B)
        
        // Matrix A (column-major): [1 4; 2 5; 3 6] (3x2)
        // A' = [1 2 3; 4 5 6] (2x3)
        double a_gemm[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        
        // Matrix B (column-major): [2 5; 3 6; 4 7] (3x2)
        double b_gemm[6] = {2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
        
        // Matrix C (column-major, will be overwritten)
        double c_gemm[4] = {1.0, 2.0, 3.0, 4.0};
        
        double alpha_gemm = 2.0;
        double beta_gemm = 0.5; // C will be scaled and added to
        
        char transa_gemm = 'T'; // Transpose A
        char transb_gemm = 'N'; // No transpose B
        
        int lda_gemm = 3; // Leading dimension of A before transpose
        int ldb_gemm = 3; // Leading dimension of B
        int ldc_gemm = 2; // Leading dimension of C
        
        std::cout << "\nMatrix A (to be transposed):" << std::endl;
        print_matrix(a_gemm, k_gemm, m_gemm, lda_gemm);
        
        std::cout << "Matrix B:" << std::endl;
        print_matrix(b_gemm, k_gemm, n_gemm, ldb_gemm);
        
        std::cout << "Matrix C before dgemm:" << std::endl;
        print_matrix(c_gemm, m_gemm, n_gemm, ldc_gemm);
        
        // Call embedded dgemm
        nasoq::embedded::dgemm(&transa_gemm, &transb_gemm, &m_gemm, &n_gemm, &k_gemm,
                             &alpha_gemm, a_gemm, &lda_gemm, b_gemm, &ldb_gemm,
                             &beta_gemm, c_gemm, &ldc_gemm);
        
        std::cout << "Matrix C after dgemm:" << std::endl;
        print_matrix(c_gemm, m_gemm, n_gemm, ldc_gemm);
        
        // Expected result: C = 2*A'*B + 0.5*C
        // C = 2*[1 2 3; 4 5 6]*[2 5; 3 6; 4 7] + 0.5*[1 3; 2 4]
        // C = 2*[20 41; 47 98] + [0.5 1.5; 1.0 2.0]
        // C = [40.5 83.5; 95.0 198.0]
        
        // Check results
        test_assert(approx_equal(c_gemm[0], 40.5), "c[0,0] = 2*(1*2 + 2*3 + 3*4) + 0.5*1 = 40.5");
        test_assert(approx_equal(c_gemm[1], 95.0), "c[1,0] = 2*(4*2 + 5*3 + 6*4) + 0.5*2 = 95.0");
        test_assert(approx_equal(c_gemm[2], 83.5), "c[0,1] = 2*(1*5 + 2*6 + 3*7) + 0.5*3 = 83.5");
        test_assert(approx_equal(c_gemm[3], 198.0), "c[1,1] = 2*(4*5 + 5*6 + 6*7) + 0.5*4 = 198.0");
    }
    
    //====================================================================
    // Test dtrsm
    //====================================================================
    begin_test("dtrsm");
    
    // Test case 1: Lower triangular
    {
        const int m_trsm = 2; // Rows of B
        const int n_trsm = 2; // Columns of B
        
        // Lower triangular matrix A (column-major): [2 0; 1 3]
        double a_trsm[4] = {2.0, 1.0, 0.0, 3.0};
        
        // Matrix B (column-major, will be overwritten): [6 18; 5 25]
        double b_trsm[4] = {6.0, 5.0, 18.0, 25.0};
        
        double alpha_trsm = 1.0;
        
        char side_trsm = 'L'; // op(A) on left of X
        char uplo_trsm = 'L'; // A is lower triangular
        char transa_trsm = 'N'; // No transpose
        char diag_trsm = 'N'; // Not unit triangular
        
        int lda_trsm = 2; // Leading dimension of A
        int ldb_trsm = 2; // Leading dimension of B
        
        std::cout << "Matrix A (lower triangular):" << std::endl;
        print_matrix(a_trsm, m_trsm, n_trsm, lda_trsm);
        
        std::cout << "Matrix B before dtrsm:" << std::endl;
        print_matrix(b_trsm, m_trsm, n_trsm, ldb_trsm);
        
        // Call embedded dtrsm
        nasoq::embedded::dtrsm(&side_trsm, &uplo_trsm, &transa_trsm, &diag_trsm,
                             &m_trsm, &n_trsm, &alpha_trsm, a_trsm, &lda_trsm, b_trsm, &ldb_trsm);
        
        std::cout << "Matrix X (solution) after dtrsm:" << std::endl;
        print_matrix(b_trsm, m_trsm, n_trsm, ldb_trsm);
        
        // Expected result: X where A*X = B
        // To solve manually:
        // [2 0] * [x11 x12] = [6  18]
        // [1 3]   [x21 x22]   [5  25]
        //
        // 2*x11 = 6 => x11 = 3
        // 1*x11 + 3*x21 = 5 => 3*x21 = 5-3 => x21 = 2/3
        // 2*x12 = 18 => x12 = 9
        // 1*x12 + 3*x22 = 25 => 3*x22 = 25-9 => x22 = 16/3
        
        // Check results
        test_assert(approx_equal(b_trsm[0], 3.0), "b_trsm[0] should be 3.0");
        test_assert(approx_equal(b_trsm[1], 2.0/3.0), "b_trsm[1] should be 2.0/3.0");
        test_assert(approx_equal(b_trsm[2], 9.0), "b_trsm[2] should be 9.0");
        test_assert(approx_equal(b_trsm[3], 16.0/3.0), "b_trsm[3] should be 16.0/3.0");
    }
    
    // Test case 2: Upper triangular
    {
        const int m_trsm = 2; // Rows of B
        const int n_trsm = 2; // Columns of B
        
        // Upper triangular matrix A (column-major): [2 1; 0 3]
        double a_trsm[4] = {2.0, 0.0, 1.0, 3.0};
        
        // Matrix B (column-major, will be overwritten): [12 24; 3 18]
        double b_trsm[4] = {12.0, 3.0, 24.0, 18.0};
        
        double alpha_trsm = 1.0;
        
        char side_trsm = 'L'; // op(A) on left of X
        char uplo_trsm = 'U'; // A is upper triangular
        char transa_trsm = 'N'; // No transpose
        char diag_trsm = 'N'; // Not unit triangular
        
        int lda_trsm = 2; // Leading dimension of A
        int ldb_trsm = 2; // Leading dimension of B
        
        std::cout << "\nMatrix A (upper triangular):" << std::endl;
        print_matrix(a_trsm, m_trsm, n_trsm, lda_trsm);
        
        std::cout << "Matrix B before dtrsm:" << std::endl;
        print_matrix(b_trsm, m_trsm, n_trsm, ldb_trsm);
        
        // Call embedded dtrsm
        nasoq::embedded::dtrsm(&side_trsm, &uplo_trsm, &transa_trsm, &diag_trsm,
                             &m_trsm, &n_trsm, &alpha_trsm, a_trsm, &lda_trsm, b_trsm, &ldb_trsm);
        
        std::cout << "Matrix X (solution) after dtrsm:" << std::endl;
        print_matrix(b_trsm, m_trsm, n_trsm, ldb_trsm);
        
        // Expected result: X where A*X = B
        // To solve manually:
        // [2 1] * [x11 x12] = [12 24]
        // [0 3]   [x21 x22]   [3  18]
        //
        // 3*x21 = 3 => x21 = 1
        // 3*x22 = 18 => x22 = 6
        // 2*x11 + 1*x21 = 12 => 2*x11 = 12-1 => x11 = 5.5
        // 2*x12 + 1*x22 = 24 => 2*x12 = 24-6 => x12 = 9
        
        // Check results
        test_assert(approx_equal(b_trsm[0], 5.5), "b_trsm[0] should be 5.5");
        test_assert(approx_equal(b_trsm[1], 1.0), "b_trsm[1] should be 1.0");
        test_assert(approx_equal(b_trsm[2], 9.0), "b_trsm[2] should be 9.0");
        test_assert(approx_equal(b_trsm[3], 6.0), "b_trsm[3] should be 6.0");
    }
    
    //====================================================================
    // Test sym_sytrf
    //====================================================================
    begin_test("sym_sytrf");
    
    // Create a 3x3 symmetric matrix A = [ 4 2 0; 2 5 1; 0 1 3 ]
    // in column-major format
    const int n_sytrf = 3;
    double a_sytrf[9] = {
        4.0, 2.0, 0.0,  // First column
        2.0, 5.0, 1.0,  // Second column
        0.0, 1.0, 3.0   // Third column
    };
    
    // Expected factorization:
    // D = diag(4, 4, 2.75)
    // L = [ 1 0 0; 0.5 1 0; 0 0.25 1 ]
    
    int stride_sytrf = n_sytrf;
    int nbpivot_sytrf = 0;
    double critere_sytrf = 1e-10;
    
    std::cout << "Original matrix A:" << std::endl;
    print_matrix(a_sytrf, n_sytrf, n_sytrf, stride_sytrf);
    
    // Call embedded sym_sytrf
    nasoq::embedded::sym_sytrf(a_sytrf, n_sytrf, stride_sytrf, &nbpivot_sytrf, critere_sytrf);
    
    std::cout << "After sym_sytrf:" << std::endl;
    print_matrix(a_sytrf, n_sytrf, n_sytrf, stride_sytrf);
    
    // Extract diagonal elements (D)
    double d_extracted[3] = {
        a_sytrf[0],        // D[0,0]
        a_sytrf[4],        // D[1,1]
        a_sytrf[8]         // D[2,2]
    };
    
    // Extract off-diagonal elements (L, with implicit unit diagonal)
    double l_extracted[3] = {
        // Unit diagonal elements are implicit and not stored
        a_sytrf[3],        // L[1,0]
        a_sytrf[6],        // L[2,0]
        a_sytrf[7]         // L[2,1]
    };
    
    // Verify factorization
    test_assert(approx_equal(d_extracted[0], 4.0), "D[0,0] should be 4.0");
    test_assert(approx_equal(d_extracted[1], 4.0), "D[1,1] should be 4.0");
    test_assert(approx_equal(d_extracted[2], 2.75), "D[2,2] should be 2.75");
    
    test_assert(approx_equal(l_extracted[0], 0.5), "L[1,0] should be 0.5");
    test_assert(approx_equal(l_extracted[1], 0.0), "L[2,0] should be 0.0");
    test_assert(approx_equal(l_extracted[2], 0.25), "L[2,1] should be 0.25");
    
    // Reconstruct original matrix to verify factorization
    // A = L * D * L^T
    double reconstructed[9] = {0.0};
    
    // Manual reconstruction for verification
    // A[0,0] = D[0,0] * L[0,0]^2 = 4.0 * 1^2 = 4.0
    reconstructed[0] = d_extracted[0];
    
    // A[1,0] = D[0,0] * L[1,0] = 4.0 * 0.5 = 2.0
    reconstructed[3] = d_extracted[0] * l_extracted[0];
    
    // A[1,1] = D[0,0] * L[1,0]^2 + D[1,1] = 4.0 * 0.5^2 + 4.0 = 5.0
    reconstructed[4] = d_extracted[0] * l_extracted[0] * l_extracted[0] + d_extracted[1];
    
    // A[2,0] = D[0,0] * L[2,0] + D[1,1] * L[2,1] * L[1,0] = 4.0 * 0.0 + 4.0 * 0.25 * 0.5 = 0.5
    reconstructed[6] = d_extracted[0] * l_extracted[1] + d_extracted[1] * l_extracted[2] * l_extracted[0];
    
    // A[2,1] = D[0,0] * L[1,0] * L[2,0] + D[1,1] * L[2,1] = 4.0 * 0.5 * 0.0 + 4.0 * 0.25 = 1.0
    reconstructed[7] = d_extracted[0] * l_extracted[0] * l_extracted[1] + d_extracted[1] * l_extracted[2];
    
    // A[2,2] = D[0,0] * L[2,0]^2 + D[1,1] * L[2,1]^2 + D[2,2] = 4.0 * 0.0^2 + 4.0 * 0.25^2 + 2.75 = 3.0
    reconstructed[8] = d_extracted[0] * l_extracted[1] * l_extracted[1] + 
                      d_extracted[1] * l_extracted[2] * l_extracted[2] + 
                      d_extracted[2];
    
    std::cout << "Reconstructed matrix from LDL^T:" << std::endl;
    print_matrix(reconstructed, n_sytrf, n_sytrf, stride_sytrf);
    
    // Verify if the reconstructed matrix matches the original
    test_assert(approx_equal(reconstructed[0], 4.0), "Reconstructed A[0,0] should be 4.0");
    test_assert(approx_equal(reconstructed[3], 2.0), "Reconstructed A[1,0] should be 2.0");
    test_assert(approx_equal(reconstructed[4], 5.0), "Reconstructed A[1,1] should be 5.0");
    test_assert(approx_equal(reconstructed[6], 0.0), "Reconstructed A[2,0] should be 0.0");
    test_assert(approx_equal(reconstructed[7], 1.0), "Reconstructed A[2,1] should be 1.0");
    test_assert(approx_equal(reconstructed[8], 3.0), "Reconstructed A[2,2] should be 3.0");
    
    //====================================================================
    // Test dlapmt (column permutation)
    //====================================================================
    begin_test("dlapmt");
    
    // Define LAPACK_ROW_MAJOR and LAPACK_COL_MAJOR constants to match clapacke.h
    const int LAPACK_ROW_MAJOR = 101;
    const int LAPACK_COL_MAJOR = 102;
    
    // Test 1: Forward permutation with column-major matrix
    {
        const int m = 3; // Number of rows
        const int n = 4; // Number of columns
        
        // Create a matrix in column-major format
        double x_col[12] = {
            1.0, 2.0, 3.0,  // First column
            4.0, 5.0, 6.0,  // Second column
            7.0, 8.0, 9.0,  // Third column
            10.0, 11.0, 12.0 // Fourth column
        };
        
        // Permutation vector (1-indexed as per LAPACK standard)
        int k[4] = {3, 1, 4, 2};
        
        std::cout << "Original column-major matrix:" << std::endl;
        print_matrix(x_col, m, n, m);
        
        // Call embedded dlapmt with forward permutation
        int forwrd = 1; // true for forward permutation
        nasoq::embedded::dlapmt(LAPACK_COL_MAJOR, forwrd, m, n, x_col, m, k);
        
        std::cout << "After forward permutation:" << std::endl;
        print_matrix(x_col, m, n, m);
        
        // Expected result after permutation:
        // Column 1 should be original column 3 (7, 8, 9)
        // Column 2 should be original column 1 (1, 2, 3)
        // Column 3 should be original column 4 (10, 11, 12)
        // Column 4 should be original column 2 (4, 5, 6)
        
        test_assert(approx_equal(x_col[0], 7.0), "x_col[0,0] should be 7.0");
        test_assert(approx_equal(x_col[1], 8.0), "x_col[1,0] should be 8.0");
        test_assert(approx_equal(x_col[2], 9.0), "x_col[2,0] should be 9.0");
        
        test_assert(approx_equal(x_col[3], 1.0), "x_col[0,1] should be 1.0");
        test_assert(approx_equal(x_col[4], 2.0), "x_col[1,1] should be 2.0");
        test_assert(approx_equal(x_col[5], 3.0), "x_col[2,1] should be 3.0");
        
        test_assert(approx_equal(x_col[6], 10.0), "x_col[0,2] should be 10.0");
        test_assert(approx_equal(x_col[7], 11.0), "x_col[1,2] should be 11.0");
        test_assert(approx_equal(x_col[8], 12.0), "x_col[2,2] should be 12.0");
        
        test_assert(approx_equal(x_col[9], 4.0), "x_col[0,3] should be 4.0");
        test_assert(approx_equal(x_col[10], 5.0), "x_col[1,3] should be 5.0");
        test_assert(approx_equal(x_col[11], 6.0), "x_col[2,3] should be 6.0");
    }
    
    // Test 2: Backward permutation with column-major matrix
    {
        const int m = 3; // Number of rows
        const int n = 4; // Number of columns
        
        // Create a matrix in column-major format
        double x_col[12] = {
            1.0, 2.0, 3.0,  // First column
            4.0, 5.0, 6.0,  // Second column
            7.0, 8.0, 9.0,  // Third column
            10.0, 11.0, 12.0 // Fourth column
        };
        
        // Permutation vector (1-indexed as per LAPACK standard)
        int k[4] = {2, 4, 1, 3};
        
        std::cout << "\nOriginal column-major matrix for backward permutation:" << std::endl;
        print_matrix(x_col, m, n, m);
        
        // Call embedded dlapmt with backward permutation
        int forwrd = 0; // false for backward permutation
        nasoq::embedded::dlapmt(LAPACK_COL_MAJOR, forwrd, m, n, x_col, m, k);
        
        std::cout << "After backward permutation:" << std::endl;
        print_matrix(x_col, m, n, m);
        
        // Expected result after permutation:
        // Column 1 should move to column 2's position (1,2,3 -> col 2)
        // Column 2 should move to column 4's position (4,5,6 -> col 4)
        // Column 3 should move to column 1's position (7,8,9 -> col 1)
        // Column 4 should move to column 3's position (10,11,12 -> col 3)
        
        test_assert(approx_equal(x_col[0], 7.0), "x_col[0,0] should be 7.0");
        test_assert(approx_equal(x_col[1], 8.0), "x_col[1,0] should be 8.0");
        test_assert(approx_equal(x_col[2], 9.0), "x_col[2,0] should be 9.0");
        
        test_assert(approx_equal(x_col[3], 1.0), "x_col[0,1] should be 1.0");
        test_assert(approx_equal(x_col[4], 2.0), "x_col[1,1] should be 2.0");
        test_assert(approx_equal(x_col[5], 3.0), "x_col[2,1] should be 3.0");
        
        test_assert(approx_equal(x_col[6], 10.0), "x_col[0,2] should be 10.0");
        test_assert(approx_equal(x_col[7], 11.0), "x_col[1,2] should be 11.0");
        test_assert(approx_equal(x_col[8], 12.0), "x_col[2,2] should be 12.0");
        
        test_assert(approx_equal(x_col[9], 4.0), "x_col[0,3] should be 4.0");
        test_assert(approx_equal(x_col[10], 5.0), "x_col[1,3] should be 5.0");
        test_assert(approx_equal(x_col[11], 6.0), "x_col[2,3] should be 6.0");
    }
    
    // Test 3: Row-major format
    {
        const int m = 3; // Number of rows
        const int n = 4; // Number of columns
        
        // Create a matrix in row-major format
        double x_row[12] = {
            1.0, 4.0, 7.0, 10.0,  // First row
            2.0, 5.0, 8.0, 11.0,  // Second row
            3.0, 6.0, 9.0, 12.0   // Third row
        };
        
        // Permutation vector (1-indexed as per LAPACK standard)
        int k[4] = {3, 1, 4, 2};
        
        std::cout << "\nOriginal row-major matrix:" << std::endl;
        print_matrix(x_row, m, n, n);
        
        // Call embedded dlapmt with forward permutation
        int forwrd = 1; // true for forward permutation
        nasoq::embedded::dlapmt(LAPACK_ROW_MAJOR, forwrd, m, n, x_row, n, k);
        
        std::cout << "After forward permutation in row-major format:" << std::endl;
        print_matrix(x_row, m, n, n);
        
        // Expected result after permutation:
        // Column 1 should be original column 3 (7, 8, 9)
        // Column 2 should be original column 1 (1, 2, 3)
        // Column 3 should be original column 4 (10, 11, 12)
        // Column 4 should be original column 2 (4, 5, 6)
        
        test_assert(approx_equal(x_row[0], 7.0), "x_row[0,0] should be 7.0");
        test_assert(approx_equal(x_row[1], 1.0), "x_row[0,1] should be 1.0");
        test_assert(approx_equal(x_row[2], 10.0), "x_row[0,2] should be 10.0");
        test_assert(approx_equal(x_row[3], 4.0), "x_row[0,3] should be 4.0");
        
        test_assert(approx_equal(x_row[4], 8.0), "x_row[1,0] should be 8.0");
        test_assert(approx_equal(x_row[5], 2.0), "x_row[1,1] should be 2.0");
        test_assert(approx_equal(x_row[6], 11.0), "x_row[1,2] should be 11.0");
        test_assert(approx_equal(x_row[7], 5.0), "x_row[1,3] should be 5.0");
        
        test_assert(approx_equal(x_row[8], 9.0), "x_row[2,0] should be 9.0");
        test_assert(approx_equal(x_row[9], 3.0), "x_row[2,1] should be 3.0");
        test_assert(approx_equal(x_row[10], 12.0), "x_row[2,2] should be 12.0");
        test_assert(approx_equal(x_row[11], 6.0), "x_row[2,3] should be 6.0");
    }
    
    //====================================================================
    // Test dgetrf
    //====================================================================
    begin_test("dgetrf");

    // Test case 1: Non-singular matrix (column-major)
    {
        const int m = 3;
        const int n = 3;
        
        // Matrix A in column-major format:
        // [ 2  -1   0 ]
        // [ 1   3   2 ]
        // [ 0   1   1 ]
        double a[9] = {
            2.0, 1.0, 0.0,  // First column
            -1.0, 3.0, 1.0, // Second column
            0.0, 2.0, 1.0   // Third column
        };
        
        int ipiv[3] = {0, 0, 0}; // Pivot indices
        
        std::cout << "Original matrix A:" << std::endl;
        print_matrix(a, m, n, m);
        
        // Call embedded dgetrf
        int info = nasoq::embedded::dgetrf(LAPACK_COL_MAJOR, m, n, a, m, ipiv);
        
        std::cout << "After dgetrf:" << std::endl;
        print_matrix(a, m, n, m);
        
        std::cout << "Pivot indices: ";
        for (int i = 0; i < n; i++) {
            std::cout << ipiv[i] << " ";
        }
        std::cout << std::endl;
        
        // Verify factorization result
        test_assert(info == 0, "LU factorization should succeed with info = 0");
        
        // Create a copy of A (L and U factors)
        double lu[9];
        for (int i = 0; i < 9; i++) {
            lu[i] = a[i];
        }
        
        // Extract L (lower triangular with unit diagonal) and U (upper triangular)
        double l[9] = {
            1.0, 0.0, 0.0,
            lu[1], 1.0, 0.0,
            lu[2], lu[5], 1.0
        };
        
        double u[9] = {
            lu[0], lu[3], lu[6],
            0.0, lu[4], lu[7],
            0.0, 0.0, lu[8]
        };
        
        // Perform L*U multiplication (without considering pivoting for simplicity)
        double result[9] = {0};
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    if (k <= i && k <= j) {
                        result[i + j*m] += l[i + k*m] * u[k + j*m];
                    }
                }
            }
        }
        
        std::cout << "Reconstructed matrix (L*U):" << std::endl;
        print_matrix(result, m, n, m);
        
        // Check that L*U (with permutation) equals the original matrix
        // This is a simplified check that doesn't fully account for permutations
        test_assert(approx_equal(result[0], 2.0), "Reconstructed[0,0] should be 2.0");
        test_assert(approx_equal(result[3], -1.0), "Reconstructed[0,1] should be -1.0");
        test_assert(approx_equal(result[6], 0.0), "Reconstructed[0,2] should be 0.0");
        test_assert(approx_equal(result[1], 1.0), "Reconstructed[1,0] should be 1.0");
        test_assert(approx_equal(result[4], 3.0), "Reconstructed[1,1] should be 3.0");
        test_assert(approx_equal(result[7], 2.0), "Reconstructed[1,2] should be 2.0");
        test_assert(approx_equal(result[2], 0.0), "Reconstructed[2,0] should be 0.0");
        test_assert(approx_equal(result[5], 1.0), "Reconstructed[2,1] should be 1.0");
        test_assert(approx_equal(result[8], 1.0), "Reconstructed[2,2] should be 1.0");
    }

    // Test case 2: Singular matrix
    {
        const int m = 3;
        const int n = 3;
        
        // Singular matrix A in column-major format:
        // [ 1  2  3 ]
        // [ 2  4  6 ]
        // [ 0  1  7 ]
        double a[9] = {
            1.0, 2.0, 0.0,  // First column
            2.0, 4.0, 1.0,  // Second column
            3.0, 6.0, 7.0   // Third column
        };
        
        int ipiv[3] = {0, 0, 0}; // Pivot indices
        
        std::cout << "\nOriginal singular matrix A:" << std::endl;
        print_matrix(a, m, n, m);
        
        // Call embedded dgetrf
        int info = nasoq::embedded::dgetrf(LAPACK_COL_MAJOR, m, n, a, m, ipiv);
        
        std::cout << "After dgetrf:" << std::endl;
        print_matrix(a, m, n, m);
        
        std::cout << "Pivot indices: ";
        for (int i = 0; i < n; i++) {
            std::cout << ipiv[i] << " ";
        }
        std::cout << std::endl;
        
        // For a singular matrix, info should be positive and indicate first zero pivot
        test_assert(info > 0, "For singular matrix, info should be positive");
    }

    // Test case 3: Row-major format
    {
        const int m = 2;
        const int n = 2;
        
        // Matrix A in row-major format:
        // [ 4  3 ]
        // [ 6  3 ]
        double a[4] = {
            4.0, 3.0,  // First row
            6.0, 3.0   // Second row
        };
        
        int ipiv[2] = {0, 0}; // Pivot indices
        
        std::cout << "\nOriginal row-major matrix A:" << std::endl;
        print_matrix(a, m, n, n);
        
        // Call embedded dgetrf
        int info = nasoq::embedded::dgetrf(LAPACK_ROW_MAJOR, m, n, a, n, ipiv);
        
        std::cout << "After dgetrf:" << std::endl;
        print_matrix(a, m, n, n);
        
        std::cout << "Pivot indices: ";
        for (int i = 0; i < n; i++) {
            std::cout << ipiv[i] << " ";
        }
        std::cout << std::endl;
        
        // Verify factorization result
        test_assert(info == 0, "LU factorization should succeed with info = 0");
    }
    
    //====================================================================
    // Test dsytrf
    //====================================================================
    begin_test("dsytrf");

    // Test case 1: Lower triangular format (column-major)
    {
        const int n = 3;
        
        // Symmetric matrix A in column-major format:
        // [ 4  0  0 ]
        // [ 2  5  0 ]
        // [ 1  3  6 ]
        double a[9] = {
            4.0, 2.0, 1.0,  // First column
            0.0, 5.0, 3.0,  // Second column
            0.0, 0.0, 6.0   // Third column
        };
        
        int ipiv[3] = {0, 0, 0}; // Pivot indices
        
        std::cout << "Original matrix A (lower triangular format):" << std::endl;
        print_matrix(a, n, n, n);
        
        // Call embedded dsytrf
        int info = nasoq::embedded::dsytrf(LAPACK_COL_MAJOR, 'L', n, a, n, ipiv);
        
        std::cout << "After dsytrf:" << std::endl;
        print_matrix(a, n, n, n);
        
        std::cout << "Pivot indices: ";
        for (int i = 0; i < n; i++) {
            std::cout << ipiv[i] << " ";
        }
        std::cout << std::endl;
        
        // Verify factorization result
        test_assert(info == 0, "DSYTRF factorization should succeed with info = 0");
        
        // Extract D and L from the factorized matrix
        // For L, the diagonal is implicitly 1
        
        // Create a copy for verification
        double l[9] = {0};
        double d[9] = {0};
        
        // Extract L (unit lower triangular) from the factorized matrix
        l[0] = 1.0; // Diagonal elements are implicitly 1
        l[4] = 1.0;
        l[8] = 1.0;
        
        l[1] = 0.0; // Upper triangular part is zero
        l[2] = 0.0;
        l[5] = 0.0;
        
        l[3] = a[1]; // Lower triangular part holds L values
        l[6] = a[2];
        l[7] = a[5];
        
        // Extract D (block diagonal) based on ipiv
        // This is a simplified extraction that doesn't fully handle 2x2 blocks
        d[0] = a[0]; // Diagonal elements hold D values
        d[4] = a[4];
        d[8] = a[8];
        
        std::cout << "L matrix:" << std::endl;
        print_matrix(l, n, n, n);
        
        std::cout << "D matrix:" << std::endl;
        print_matrix(d, n, n, n);
        
        // Multiply L*D*L^T to reconstruct the original matrix
        // This is a simplified multiplication for verification
        double result[9] = {0};
        
        // First compute L*D
        double ld[9] = {0};
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    ld[i + j*n] += l[i + k*n] * d[k + j*n];
                }
            }
        }
        
        // Then compute (L*D)*L^T
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    result[i + j*n] += ld[i + k*n] * l[j + k*n]; // Note: L^T[k,j] = L[j,k]
                }
            }
        }
        
        std::cout << "Reconstructed matrix (L*D*L^T):" << std::endl;
        print_matrix(result, n, n, n);
        
        // Verify the reconstruction against the original matrix
        test_assert(approx_equal(result[0], 4.0), "Reconstructed[0,0] should be 4.0");
        test_assert(approx_equal(result[1], 2.0), "Reconstructed[1,0] should be 2.0");
        test_assert(approx_equal(result[2], 1.0), "Reconstructed[2,0] should be 1.0");
        test_assert(approx_equal(result[3], 2.0), "Reconstructed[0,1] should be 2.0");
        test_assert(approx_equal(result[4], 5.0), "Reconstructed[1,1] should be 5.0");
        test_assert(approx_equal(result[5], 3.0), "Reconstructed[2,1] should be 3.0");
        test_assert(approx_equal(result[6], 1.0), "Reconstructed[0,2] should be 1.0");
        test_assert(approx_equal(result[7], 3.0), "Reconstructed[1,2] should be 3.0");
        test_assert(approx_equal(result[8], 6.0), "Reconstructed[2,2] should be 6.0");
    }

    // Test case 2: Upper triangular format (column-major)
    {
        const int n = 3;
        
        // Symmetric matrix A in column-major format:
        // [ 4  2  1 ]
        // [ 0  5  3 ]
        // [ 0  0  6 ]
        double a[9] = {
            4.0, 0.0, 0.0,  // First column
            2.0, 5.0, 0.0,  // Second column
            1.0, 3.0, 6.0   // Third column
        };
        
        int ipiv[3] = {0, 0, 0}; // Pivot indices
        
        std::cout << "\nOriginal matrix A (upper triangular format):" << std::endl;
        print_matrix(a, n, n, n);
        
        // Call embedded dsytrf
        int info = nasoq::embedded::dsytrf(LAPACK_COL_MAJOR, 'U', n, a, n, ipiv);
        
        std::cout << "After dsytrf:" << std::endl;
        print_matrix(a, n, n, n);
        
        std::cout << "Pivot indices: ";
        for (int i = 0; i < n; i++) {
            std::cout << ipiv[i] << " ";
        }
        std::cout << std::endl;
        
        // Verify factorization result
        test_assert(info == 0, "DSYTRF factorization should succeed with info = 0");
    }

    // Test case 3: Row-major format
    {
        const int n = 3;
        
        // Symmetric matrix A in row-major format:
        // [ 4  2  1 ]
        // [ 2  5  3 ]
        // [ 1  3  6 ]
        double a[9] = {
            4.0, 2.0, 1.0,  // First row
            2.0, 5.0, 3.0,  // Second row
            1.0, 3.0, 6.0   // Third row
        };
        
        int ipiv[3] = {0, 0, 0}; // Pivot indices
        
        std::cout << "\nOriginal matrix A (row-major format):" << std::endl;
        print_matrix(a, n, n, n);
        
        // Call embedded dsytrf
        int info = nasoq::embedded::dsytrf(LAPACK_ROW_MAJOR, 'L', n, a, n, ipiv);
        
        std::cout << "After dsytrf:" << std::endl;
        print_matrix(a, n, n, n);
        
        std::cout << "Pivot indices: ";
        for (int i = 0; i < n; i++) {
            std::cout << ipiv[i] << " ";
        }
        std::cout << std::endl;
        
        // Verify factorization result
        test_assert(info == 0, "DSYTRF factorization should succeed with info = 0");
    }
    
    std::cout << "\nEmbedded BLAS functions test completed." << std::endl;
    std::cout << "=================== TEST SUMMARY ===================" << std::endl;
    
    int total_passed = 0;
    int total_failed = 0;
    bool all_passed = true;
    
    // Replace structured bindings with traditional iterator approach
    for (const auto& pair : test_functions) {
        const std::string& name = pair.first;
        const TestFunction& func = pair.second;
        
        total_passed += func.passed;
        total_failed += func.failed;
        if (func.failed > 0) {
            all_passed = false;
        }
        
        // Print test status
        std::cout << std::left << std::setw(25) << name << ": ";
        if (func.failed == 0) {
            std::cout << "PASSED (" << func.passed << " checks)" << std::endl;
        } else {
            std::cout << "FAILED (" << func.passed << " passed, " << func.failed << " failed)" << std::endl;
            for (const auto& failure : func.failures) {
                std::cout << "  - " << failure << std::endl;
            }
        }
    }
    
    std::cout << "===================================================" << std::endl;
    std::cout << "OVERALL: " << (all_passed ? "PASSED" : "FAILED") << " (Total: " 
              << total_passed << " passed, " << total_failed << " failed)" << std::endl;
    
    return all_passed ? 0 : 1;
} 