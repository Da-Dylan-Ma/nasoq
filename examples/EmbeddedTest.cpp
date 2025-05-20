#include <iostream>
#include <cassert>
#include <iomanip>
#include "nasoq/embedded/embedded_blas.h"

// Helper function to check if two values are approximately equal
bool approx_equal(double a, double b, double epsilon = 1e-10) {
    return std::abs(a - b) < epsilon;
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
    std::cout << "\n===== Testing dscal =====" << std::endl;
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
    assert(approx_equal(x[0], 2.0));
    assert(approx_equal(x[1], 4.0));
    assert(approx_equal(x[2], 6.0));
    assert(approx_equal(x[3], 8.0));
    assert(approx_equal(x[4], 10.0));
    
    //====================================================================
    // Test dsyr
    //====================================================================
    std::cout << "\n===== Testing dsyr =====" << std::endl;
    const int m = 3;
    double a[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}; // 3x3 identity matrix
    double y[3] = {1.0, 2.0, 3.0};
    const double beta = 2.0; // Use 2.0 to match EMBEDDED_Test.cpp tests
    const char uplo = 'L';
    int incy = 1;
    int lda = 3;
    
    std::cout << "Original matrix a:" << std::endl;
    print_matrix(a, m, m, lda);
    
    // Call embedded dsyr
    nasoq::embedded::dsyr(&uplo, &m, &beta, y, &incy, a, &lda);
    
    std::cout << "After dsyr (rank-1 update with y and beta=2.0):" << std::endl;
    print_matrix(a, m, m, lda);
    
    // Verify results (same as in EMBEDDED_Test.cpp)
    assert(approx_equal(a[0], 3.0));  // a[0,0]
    assert(approx_equal(a[3], 2.0));  // a[1,0]
    assert(approx_equal(a[4], 9.0));  // a[1,1]
    assert(approx_equal(a[6], 3.0));  // a[2,0]
    assert(approx_equal(a[7], 6.0));  // a[2,1]
    assert(approx_equal(a[8], 19.0)); // a[2,2]
    
    //====================================================================
    // Test dcopy
    //====================================================================
    std::cout << "\n===== Testing dcopy =====" << std::endl;
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
        assert(approx_equal(dst[i], src[i]));
    }
    
    //====================================================================
    // Test blocked_2by2_solver
    //====================================================================
    std::cout << "\n===== Testing blocked_2by2_solver =====" << std::endl;
    
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
    assert(approx_equal(sol_check1, 4.0));
    assert(approx_equal(sol_check2, 11.0));
    
    // For 1x1 block: 4.0 * rhs[2] = 12.0
    assert(approx_equal(D[2] * rhs[2], 12.0));
    
    //====================================================================
    // Test dgemv
    //====================================================================
    std::cout << "\n===== Testing dgemv =====" << std::endl;
    
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
        assert(approx_equal(vec_y[0], 15.0));
        assert(approx_equal(vec_y[1], 20.0));
        assert(approx_equal(vec_y[2], 25.0));
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
        assert(approx_equal(vec_y[0], 40.5));
        assert(approx_equal(vec_y[1], 94.5));
    }
    
    //====================================================================
    // Test blocked_2by2_mult
    //====================================================================
    std::cout << "\n===== Testing blocked_2by2_mult =====" << std::endl;
    
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
        assert(approx_equal(dst_mult[0], 2.0));
        assert(approx_equal(dst_mult[1], 12.0));
        assert(approx_equal(dst_mult[2], 4.0));
        assert(approx_equal(dst_mult[3], 15.0));
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
        assert(approx_equal(dst_mult[0], 6.0));
        assert(approx_equal(dst_mult[1], 13.0));
        assert(approx_equal(dst_mult[2], 9.0));
        assert(approx_equal(dst_mult[3], 17.0));
    }
    
    //====================================================================
    // Test dgemm
    //====================================================================
    std::cout << "\n===== Testing dgemm =====" << std::endl;
    
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
        assert(approx_equal(c_gemm[0], 7.0));  // c[0,0]
        assert(approx_equal(c_gemm[1], 10.0)); // c[1,0]
        assert(approx_equal(c_gemm[2], 15.0)); // c[0,1]
        assert(approx_equal(c_gemm[3], 22.0)); // c[1,1]
        assert(approx_equal(c_gemm[4], 23.0)); // c[0,2]
        assert(approx_equal(c_gemm[5], 34.0)); // c[1,2]
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
        assert(approx_equal(c_gemm[0], 40.5));  // c[0,0]
        assert(approx_equal(c_gemm[1], 95.0));  // c[1,0]
        assert(approx_equal(c_gemm[2], 83.5));  // c[0,1]
        assert(approx_equal(c_gemm[3], 198.0)); // c[1,1]
    }
    
    //====================================================================
    // Test dtrsm
    //====================================================================
    std::cout << "\n===== Testing dtrsm =====" << std::endl;
    
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
        assert(approx_equal(b_trsm[0], 3.0));      // x11
        assert(approx_equal(b_trsm[1], 2.0/3.0));  // x21
        assert(approx_equal(b_trsm[2], 9.0));      // x12
        assert(approx_equal(b_trsm[3], 16.0/3.0)); // x22
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
        assert(approx_equal(b_trsm[0], 5.5)); // x11
        assert(approx_equal(b_trsm[1], 1.0)); // x21
        assert(approx_equal(b_trsm[2], 9.0)); // x12
        assert(approx_equal(b_trsm[3], 6.0)); // x22
    }
    
    std::cout << "\nEmbedded BLAS functions test completed successfully!" << std::endl;
    return 0;
} 