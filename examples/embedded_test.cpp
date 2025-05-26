#include <iostream>
#include <cassert>
#include <iomanip>
#include <map>
#include <string>
#include <vector>
#include <cmath>  // For std::isnan and std::isinf
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
    std::cout << "\n===== STARTING TEST: " << name << " =====" << std::endl;
    
    // Initialize if not already present
    if (test_functions.find(name) == test_functions.end()) {
        test_functions[name] = TestFunction{name};
    }
}

// End test function with summary
void end_test() {
    TestFunction& func = test_functions[current_test_function];
    std::cout << "===== COMPLETED TEST: " << current_test_function << " =====" << std::endl;
    std::cout << "  Passed: " << func.passed << ", Failed: " << func.failed << std::endl;
    
    if (func.failed > 0) {
        std::cout << "  Failures:" << std::endl;
        for (const auto& failure : func.failures) {
            std::cout << "    - " << failure << std::endl;
        }
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
    std::cout << "This test will exercise all implemented embedded BLAS/LAPACK functions" << std::endl;
    std::cout << "============================================================" << std::endl;
    
    //====================================================================
    // Test dscal
    //====================================================================
    begin_test("dscal");
    {
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
    }
    end_test();
    
    //====================================================================
    // Test dsyr
    //====================================================================
    begin_test("dsyr");
    {
    const int n = 3;

    /* --- identity matrix, column-major --- */
    double a[9] = {
        1.0, 0.0, 0.0,   // column 0
        0.0, 1.0, 0.0,   // column 1
        0.0, 0.0, 1.0    // column 2
    };

    /* x = [1, 2, 3]  */
    double x[3] = {1.0, 2.0, 3.0};

    double alpha = 2.0;   // rank-1 scale
    char   uplo  = 'L';   // update lower part
    int    incx  = 1;
    int    lda   = n;

    std::cout << "Original matrix A:\n";
    print_matrix(a, n, n, lda);

    /* call embedded dsyr */
    nasoq::embedded::dsyr(&uplo, &n, &alpha, x, &incx, a, &lda);

    std::cout << "After dsyr (alpha = 2, lower):\n";
    print_matrix(a, n, n, lda);

    /* expected lower triangle:
        [ 3  0  0
            4  9  0
            6 12 19 ]                               */

    test_assert(approx_equal(a[0], 3.0),  "A(0,0) = 3");
    test_assert(approx_equal(a[1], 4.0),  "A(1,0) = 4");
    test_assert(approx_equal(a[2], 6.0),  "A(2,0) = 6");
    test_assert(approx_equal(a[4], 9.0),  "A(1,1) = 9");
    test_assert(approx_equal(a[5], 12.0), "A(2,1) = 12");
    test_assert(approx_equal(a[8], 19.0), "A(2,2) = 19");
    }
    end_test();
    
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
    
    end_test();
    
    //====================================================================
    // Test blocked_2by2_solver
    //====================================================================
    begin_test("blocked_2by2_solver");

    /* ------------------------------------------------------------------ */
    /* Storage reminder (lda_d = n = 3):
            index : 0   1   2 | 3   4   5
                    -------------------------
            value : D0  D1  D2 | L0  L1  L2
                    (diagonals) | (sub-diagonals)
    L_i is the coupling between rows i and i+1.                       */
    /* ------------------------------------------------------------------ */

    /* ============ regular case:  one 2×2 block (rows 0–1) + one 1×1 ==== */
    {
        int n     = 3;
        int n_rhs = 2;
        int lda   = n;          // column-major RHS
        int lda_d = n;          // stride to sub-diagonal

        /* 2×2 block for rows 0–1 :  [2 1; 1 3]
        1×1 block for row 2    :  [4]                                   */
        double D[6]   = { 2.0, 3.0, 4.0,
                        1.0, 0.0, 0.0 };      // L0 = 1.0 (index 3)

        /* RHS (column-major, 3×2):
            [ 4  6
            11 22
            12 24 ]                                                  */
        double rhs[6] = { 4.0, 11.0, 12.0,
                        6.0, 22.0, 24.0 };

        std::cout << "Original RHS:\n";
        print_matrix(rhs, n, n_rhs, lda);

        /* keep a copy for verification */
        double rhs_orig[6];
        std::copy(rhs, rhs + 6, rhs_orig);

        nasoq::embedded::blocked_2by2_solver(n, D, rhs, n_rhs, lda, lda_d);

        std::cout << "Solution:\n";
        print_matrix(rhs, n, n_rhs, lda);

        /* ---------- verify -------------------------------------------------- */
        auto verify_2x2 = [&](int col)
        {
            double x0 = rhs[0 + col * lda];
            double x1 = rhs[1 + col * lda];
            double b0 = 2.0 * x0 + 1.0 * x1;   // [2 1]·[x0 x1]^T
            double b1 = 1.0 * x0 + 3.0 * x1;   // [1 3]
            test_assert(approx_equal(b0, rhs_orig[0 + col * lda]),
                        "Row 0 back-multiply (col " + std::to_string(col) + ")");
            test_assert(approx_equal(b1, rhs_orig[1 + col * lda]),
                        "Row 1 back-multiply (col " + std::to_string(col) + ")");
        };
        auto verify_1x1 = [&](int col)
        {
            double x2 = rhs[2 + col * lda];
            double b2 = 4.0 * x2;
            test_assert(approx_equal(b2, rhs_orig[2 + col * lda]),
                        "Row 2 back-multiply (col " + std::to_string(col) + ")");
        };

        for (int c = 0; c < n_rhs; ++c)
        {
            verify_2x2(c);
            verify_1x1(c);
        }
    }

    /* ============ near-singular diagonal test ============================= */
    {
        int n     = 3;
        int n_rhs = 2;
        int lda   = n;
        int lda_d = n;

        double D[6]   = { 1.0e-13, 2.0, 3.0,
                        0.0,     0.0, 0.0 };   // only 1×1 blocks here

        double rhs[6] = { 1.0, 2.0, 3.0,
                        4.0, 5.0, 6.0 };

        nasoq::embedded::blocked_2by2_solver(n, D, rhs, n_rhs, lda, lda_d);

        /* just check that no NaN/Inf crept in */
        for (double v : rhs)
            test_assert(!std::isnan(v) && !std::isinf(v),
                        "near-singular solve should not yield NaN/Inf");
    }

    end_test();

    
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
    
    end_test();
    
    //====================================================================
    // Test blocked_2by2_mult
    //====================================================================
    begin_test("blocked_2by2_mult");

    /* ---------- Test case 1: only 1×1 blocks -------------------------------- */
    {
        int n_blk_mult = 2;
        int m_blk_mult = 2;

        /* D: diag[2, 3]  — no sub-diagonal ⇒ all 1×1 blocks */
        double D_mult[4] = {2.0, 3.0, 0.0, 0.0};

        /* src, column-major:  [1 2; 4 5] */
        double src_mult[4] = {1.0, 4.0, 2.0, 5.0};

        double dst_mult[4] = {0.0, 0.0, 0.0, 0.0};

        int ld_src = n_blk_mult;   // 2
        int ld_d   = 2;

        std::cout << "Diagonal D: ";
        for (int i = 0; i < n_blk_mult; ++i) std::cout << D_mult[i] << ' ';
        std::cout << "\nSource matrix:\n";
        for (int r = 0; r < n_blk_mult; ++r) {
            for (int c = 0; c < m_blk_mult; ++c)
                std::cout << src_mult[r + c * ld_src] << ' ';
            std::cout << '\n';
        }

        nasoq::embedded::blocked_2by2_mult(n_blk_mult, m_blk_mult,
                                        D_mult, src_mult, dst_mult,
                                        ld_src, ld_d);

        std::cout << "Result matrix:\n";
        for (int r = 0; r < n_blk_mult; ++r) {
            for (int c = 0; c < m_blk_mult; ++c)
                std::cout << dst_mult[r * m_blk_mult + c] << ' ';
            std::cout << '\n';
        }

        /* Expected row-major layout: [ 2, 4, 12, 15 ] */
        test_assert(approx_equal(dst_mult[0],  2.0), "dst[0] = 2");
        test_assert(approx_equal(dst_mult[1],  4.0), "dst[1] = 4");
        test_assert(approx_equal(dst_mult[2], 12.0), "dst[2] = 12");
        test_assert(approx_equal(dst_mult[3], 15.0), "dst[3] = 15");
    }

    /* ---------- Test case 2: one 2×2 block ---------------------------------- */
    {
        int n_blk_mult = 2;
        int m_blk_mult = 2;

        /* D = [2 1; 1 3]  → sub-diagonal entry == 1 */
        double D_mult[3] = {2.0, 3.0, 1.0};

        /* same src as above */
        double src_mult[4] = {1.0, 4.0, 2.0, 5.0};
        double dst_mult[4] = {0.0, 0.0, 0.0, 0.0};

        int ld_src = n_blk_mult;   // 2
        int ld_d   = 2;

        std::cout << "\nBlock diagonal D = [ 2 1 ; 1 3 ]\n"
                    "Source matrix:\n";
        for (int r = 0; r < n_blk_mult; ++r) {
            for (int c = 0; c < m_blk_mult; ++c)
                std::cout << src_mult[r + c * ld_src] << ' ';
            std::cout << '\n';
        }

        nasoq::embedded::blocked_2by2_mult(n_blk_mult, m_blk_mult,
                                        D_mult, src_mult, dst_mult,
                                        ld_src, ld_d);

        std::cout << "Result matrix:\n";
        for (int r = 0; r < n_blk_mult; ++r) {
            for (int c = 0; c < m_blk_mult; ++c)
                std::cout << dst_mult[r * m_blk_mult + c] << ' ';
            std::cout << '\n';
        }

        /* Expected row-major layout: [ 6, 9, 13, 17 ] */
        test_assert(approx_equal(dst_mult[0],  6.0), "dst[0] = 6");
        test_assert(approx_equal(dst_mult[1],  9.0), "dst[1] = 9");
        test_assert(approx_equal(dst_mult[2], 13.0), "dst[2] = 13");
        test_assert(approx_equal(dst_mult[3], 17.0), "dst[3] = 17");
    }

    end_test();
    
    //====================================================================
    // Test dgemm
    //====================================================================
    begin_test("dgemm");

    // ---------- Test case 1: No transpose ---------------------------------
    {
        const int m_gemm = 2;                 // rows of C and A
        const int n_gemm = 3;                 // cols of C and B
        const int k_gemm = 2;                 // cols of A, rows of B

        // A (col-major): [1 3; 2 4]
        double a_gemm[4] = {1.0, 2.0, 3.0, 4.0};

        // B (col-major): [1 3 5; 2 4 6]
        double b_gemm[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

        // C (col-major) – initial values will be overwritten
        double c_gemm[6] = {1.0, 2.0, 4.0, 5.0, 7.0, 8.0};

        double alpha_gemm = 1.0;
        double beta_gemm  = 0.0;

        char transa_gemm = 'N';
        char transb_gemm = 'N';

        int lda_gemm = 2;
        int ldb_gemm = 2;
        int ldc_gemm = 2;

        nasoq::embedded::dgemm(&transa_gemm, &transb_gemm,
                            &m_gemm, &n_gemm, &k_gemm,
                            &alpha_gemm, a_gemm, &lda_gemm,
                            b_gemm, &ldb_gemm, &beta_gemm,
                            c_gemm, &ldc_gemm);

        // Expected C = A*B = [[ 7 15 23 ];
        //                     [10 22 34 ]]
        test_assert(approx_equal(c_gemm[0],  7.0), "c[0,0]");
        test_assert(approx_equal(c_gemm[1], 10.0), "c[1,0]");
        test_assert(approx_equal(c_gemm[2], 15.0), "c[0,1]");
        test_assert(approx_equal(c_gemm[3], 22.0), "c[1,1]");
        test_assert(approx_equal(c_gemm[4], 23.0), "c[0,2]");
        test_assert(approx_equal(c_gemm[5], 34.0), "c[1,2]");
    }

    // ---------- Test case 2: A transposed ---------------------------------
    {
        const int m_gemm = 2;                 // rows of C and A'
        const int n_gemm = 2;                 // cols of C and B
        const int k_gemm = 3;                 // cols of A', rows of B

        // A (3×2, col-major): [1 4; 2 5; 3 6]
        // A' is 2×3
        double a_gemm[6] = {1.0, 2.0, 3.0,
                            4.0, 5.0, 6.0};

        // B (3×2, col-major): [2 5; 3 6; 4 7]
        double b_gemm[6] = {2.0, 3.0, 4.0,
                            5.0, 6.0, 7.0};

        // C (2×2, col-major) – will be scaled by β and added
        double c_gemm[4] = {1.0, 2.0,
                            3.0, 4.0};

        double alpha_gemm = 2.0;
        double beta_gemm  = 0.5;

        char transa_gemm = 'T';               // transpose A
        char transb_gemm = 'N';               // B as is

        int lda_gemm = 3;                     // leading dim of original A
        int ldb_gemm = 3;
        int ldc_gemm = 2;

        nasoq::embedded::dgemm(&transa_gemm, &transb_gemm,
                            &m_gemm, &n_gemm, &k_gemm,
                            &alpha_gemm, a_gemm, &lda_gemm,
                            b_gemm, &ldb_gemm, &beta_gemm,
                            c_gemm, &ldc_gemm);

        /*
        Expected result:
            A'B =
            [20 38;
                47 92]

            C = 2 * A'B + 0.5 * C_initial
            = [[40.5, 77.5],
                [95.0, 186.0]]
        */
        test_assert(approx_equal(c_gemm[0], 40.5), "c[0,0]");
        test_assert(approx_equal(c_gemm[1], 95.0), "c[1,0]");
        test_assert(approx_equal(c_gemm[2], 77.5), "c[0,1]");
        test_assert(approx_equal(c_gemm[3], 186.0), "c[1,1]");
    }

    end_test();
    
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
    
    end_test();
    
    //====================================================================
    // Test sym_sytrf
    //====================================================================
    begin_test("sym_sytrf");

    /*  A = [ 4 2 0 ;
            2 5 1 ;
            0 1 3 ]     (column-major)                                  */
    const int n  = 3;
    const int ld = n;
    double A[ld*n] = {
        4, 2, 0,      // col 0
        2, 5, 1,      // col 1
        0, 1, 3       // col 2
    };
    double A_orig[ld*n];
    std::copy(std::begin(A), std::end(A), A_orig);

    int nbpivot = 0;
    double crit  = 1e-10;

    nasoq::embedded::sym_sytrf(A, n, ld, &nbpivot, crit);

    /* ---------- extract D and strict lower part of L ---------------------- */
    double D[3]   = { A[0], A[4], A[8] };   // diagonal
    double L10    = A[1];   // A(1,0)
    double L20    = A[2];   // A(2,0)  (zero here)
    double L21    = A[5];   // A(2,1)

    /* checks on individual entries */
    test_assert(approx_equal(D[0], 4.0),   "D(0,0) = 4");
    test_assert(approx_equal(D[1], 4.0),   "D(1,1) = 4");
    test_assert(approx_equal(D[2], 2.75), "D(2,2) = 2.75");

    test_assert(approx_equal(L10, 0.5),  "L(1,0) = 0.5");
    test_assert(approx_equal(L20, 0.0),  "L(2,0) = 0.0");
    test_assert(approx_equal(L21, 0.25), "L(2,1) = 0.25");

    /* ---------- rebuild A = L·D·Lᵀ --------------------------------------- */
    auto L = [&](int r, int c) -> double {        // unit-lower accessor
        if (r == c) return 1.0;
        if (r == 1 && c == 0) return L10;
        if (r == 2 && c == 0) return L20;
        if (r == 2 && c == 1) return L21;
        return 0.0;
    };

    double Arec[9] = {0.0};

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                Arec[i + j*ld] += L(i,k) * D[k] * L(j,k);   // column-major write

    /* compare reconstruction with original */
    for (int idx = 0; idx < 9; ++idx)
        test_assert(approx_equal(Arec[idx], A_orig[idx]),
                    "Reconstruction mismatch at index " + std::to_string(idx));

    end_test();
    
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
    
    end_test();
    
    //====================================================================
    // Test dgetrf
    //====================================================================
    begin_test("dgetrf");

    // Test case 1: Non-singular matrix (column-major)
    {
        const int m = 3, n = 3, lda = m;

        /* A  = [ 2 −1 0 ;
                1  3 2 ;
                0  1 1 ]   (column-major) */
        double A[9] = { 2, 1, 0,
                    -1, 3, 1,
                        0, 2, 1 };

        int ipiv[3] = {0};

        std::cout << "Original A:\n";
        print_matrix(A, m, n, lda);

        double A_orig[9];
        std::copy(std::begin(A), std::end(A), A_orig);

        int info = nasoq::embedded::dgetrf(LAPACK_COL_MAJOR, m, n, A, lda, ipiv);

        std::cout << "LU factors stored in A:\n";
        print_matrix(A, m, n, lda);
        test_assert(info == 0, "dgetrf returned info = 0");

        /* ---------- build L and U --------------------------------------- */
        double L[9] = {0}, U[9] = {0};

        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                if (i > j)          L[i + j*lda] = A[i + j*lda];   // strict lower
                else                U[i + j*lda] = A[i + j*lda];   // upper incl diag
            }
        L[0] = L[4] = L[8] = 1.0;          // unit diagonal

        /* ---------- apply row pivots to the original matrix ------------- */
        double PA[9];
        std::copy(std::begin(A_orig), std::end(A_orig), PA);
        for (int k = 0; k < n; ++k) {
            int p = ipiv[k] - 1;           // 0-based pivot row
            if (p != k)
                for (int j = 0; j < n; ++j)
                    std::swap(PA[k + j*lda], PA[p + j*lda]);
        }

        /* ---------- compute L*U ----------------------------------------- */
        double LU[9] = {0};
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                for (int k = 0; k < n; ++k)
                    LU[i + j*lda] += L[i + k*lda] * U[k + j*lda];

        std::cout << "P*A  (original after pivots):\n";
        print_matrix(PA, m, n, lda);
        std::cout << "L*U  (reconstruction):\n";
        print_matrix(LU, m, n, lda);

        /* ---------- element-wise comparison ----------------------------- */
        for (int idx = 0; idx < 9; ++idx)
            test_assert(approx_equal(LU[idx], PA[idx]),
                        "Mismatch at idx " + std::to_string(idx));
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
    
    end_test();
    
    //====================================================================
    // Test dsytrf
    //====================================================================
    begin_test("dsytrf");

    // Test case 1: Lower triangular format (column-major)
    {
        const int n = 3;

        /* A (lower part stored) = [ 4 0 0 ;
                                    2 5 0 ;
                                    1 3 6 ] */
        double a[9] = {
            4.0, 2.0, 1.0,
            0.0, 5.0, 3.0,
            0.0, 0.0, 6.0
        };

        int ipiv[3] = {0, 0, 0};

        std::cout << "Original matrix A (lower-triangular format):\n";
        print_matrix(a, n, n, n);

        int info = nasoq::embedded::dsytrf(LAPACK_COL_MAJOR, 'L', n, a, n, ipiv);

        std::cout << "After dsytrf:\n";
        print_matrix(a, n, n, n);

        test_assert(info == 0, "dsytrf should return info = 0");

        /* ---------- extract L (unit-lower) and D from the packed result ------ */
        double L[9] = {0.0};
        double D[9] = {0.0};

        for (int i = 0; i < n; ++i) {
            /* unit diagonal */
            L[i + i*n] = 1.0;
            D[i + i*n] = a[i + i*n];          // diagonal element is D(i)

            for (int j = 0; j < i; ++j)       // strict lower part → L
                L[i + j*n] = a[i + j*n];
        }

        std::cout << "Extracted L:\n";
        print_matrix(L, n, n, n);
        std::cout << "Extracted D:\n";
        print_matrix(D, n, n, n);

        /* ---------- reconstruct A = L * D * Lᵀ ------------------------------ */
        double LD[9] = {0.0};
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                for (int k = 0; k < n; ++k)
                    LD[i + j*n] += L[i + k*n] * D[k + k*n] * (k == j);

        double Arec[9] = {0.0};
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                for (int k = 0; k < n; ++k)
                    Arec[i + j*n] += LD[i + k*n] * L[j + k*n];   // Lᵀ uses (j,k)

        std::cout << "Reconstructed A (L*D*Lᵀ):\n";
        print_matrix(Arec, n, n, n);

        /* ---------- compare with original lower-stored A --------------------- */
        const double ref[9] = {
            4.0, 2.0, 1.0,
            2.0, 5.0, 3.0,
            1.0, 3.0, 6.0
        };

        for (int idx = 0; idx < 9; ++idx)
            test_assert(approx_equal(Arec[idx], ref[idx]),
                        "Reconstruction mismatch at idx " + std::to_string(idx));
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
        
        int ipiv[3] = {0, 0, 0};
        
        std::cout << "\nOriginal matrix A (upper triangular format):" << std::endl;
        print_matrix(a, n, n, n);
        
        int info = nasoq::embedded::dsytrf(LAPACK_COL_MAJOR, 'U', n, a, n, ipiv);
        
        std::cout << "After dsytrf:" << std::endl;
        print_matrix(a, n, n, n);
        
        std::cout << "Pivot indices: ";
        for (int i = 0; i < n; i++) {
            std::cout << ipiv[i] << " ";
        }
        std::cout << std::endl;
        
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
        
        int ipiv[3] = {0, 0, 0};
        
        std::cout << "\nOriginal matrix A (row-major format):" << std::endl;
        print_matrix(a, n, n, n);
        
        int info = nasoq::embedded::dsytrf(LAPACK_ROW_MAJOR, 'L', n, a, n, ipiv);
        
        std::cout << "After dsytrf:" << std::endl;
        print_matrix(a, n, n, n);
        
        std::cout << "Pivot indices: ";
        for (int i = 0; i < n; i++) {
            std::cout << ipiv[i] << " ";
        }
        std::cout << std::endl;
        
        test_assert(info == 0, "DSYTRF factorization should succeed with info = 0");
    }
    
    end_test();
    
    std::cout << "\nEmbedded BLAS functions test completed." << std::endl;
    std::cout << "============================================================" << std::endl;
    
    // Print overall summary
    int total_passed = 0;
    int total_failed = 0;
    
    std::cout << "\nSUMMARY OF TEST RESULTS:" << std::endl;
    std::cout << "------------------------" << std::endl;
    
    for (const auto& pair : test_functions) {
        const TestFunction& func = pair.second;
        std::cout << func.name << ": " 
                  << func.passed << " passed, " 
                  << func.failed << " failed" << std::endl;
        
        total_passed += func.passed;
        total_failed += func.failed;
    }
    
    std::cout << "------------------------" << std::endl;
    std::cout << "TOTAL: " << total_passed << " passed, " << total_failed << " failed" << std::endl;
    
    return total_failed > 0 ? 1 : 0;  // Return non-zero if any tests failed
} 