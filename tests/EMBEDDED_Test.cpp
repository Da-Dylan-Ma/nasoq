#include <catch2/catch.hpp>
#include <nasoq/embedded/embedded_blas.h>
#include <iostream>

TEST_CASE("Direct test of embedded BLAS implementation", "[embedded]") {
    // Test dscal function
    SECTION("Test dscal") {
        const int n = 5;
        double alpha = 2.0;
        double x[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
        int incx = 1;

        std::cout << "Testing embedded dscal directly..." << std::endl;
        
        // Call our embedded implementation
        nasoq::embedded::dscal(&n, &alpha, x, &incx);
        
        // Check results
        REQUIRE(x[0] == 2.0);
        REQUIRE(x[1] == 4.0);
        REQUIRE(x[2] == 6.0);
        REQUIRE(x[3] == 8.0);
        REQUIRE(x[4] == 10.0);
    }
    
    // Test dsyr function
    SECTION("Test dsyr") {
        const int n = 3;
        double alpha = 2.0;
        double x[3] = {1.0, 2.0, 3.0};
        int incx = 1;
        double a[9] = {1.0, 0.0, 0.0, 
                       0.0, 1.0, 0.0, 
                       0.0, 0.0, 1.0}; // Initialize with identity matrix
        int lda = 3;
        char uplo = 'L';

        std::cout << "Testing embedded dsyr directly..." << std::endl;
        
        // Call our embedded implementation
        nasoq::embedded::dsyr(&uplo, &n, &alpha, x, &incx, a, &lda);
        
        // Expected result: a += 2.0 * x * x^T (lower triangular)
        // [1 0 0] + 2 * [1] * [1 2 3] = [3 0 0]
        // [0 1 0]       [2]             [2 9 0]
        // [0 0 1]       [3]             [3 6 19]
        
        // Check results (only lower triangular part matters)
        REQUIRE(a[0] == 3.0);  // a[0,0]
        REQUIRE(a[3] == 2.0);  // a[1,0]
        REQUIRE(a[4] == 9.0);  // a[1,1]
        REQUIRE(a[6] == 3.0);  // a[2,0]
        REQUIRE(a[7] == 6.0);  // a[2,1]
        REQUIRE(a[8] == 19.0); // a[2,2]
    }
    
    // Test dgemv function
    SECTION("Test dgemv (non-transpose)") {
        const int m = 3; // Rows
        const int n = 2; // Columns
        double a[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // Column-major: [1 4; 2 5; 3 6]
        double x[2] = {2.0, 3.0};
        double y[3] = {1.0, 1.0, 1.0}; // Initial values
        double alpha = 1.0;
        double beta = 1.0;
        char trans = 'N'; // No transpose
        int lda = 3;
        int incx = 1;
        int incy = 1;

        std::cout << "Testing embedded dgemv (non-transpose) directly..." << std::endl;
        
        // Call our embedded implementation
        nasoq::embedded::dgemv(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
        
        // Expected result: y = alpha*A*x + beta*y
        // [1 4]   [2]   [1]   [1+2+12]   [15]
        // [2 5] * [3] + [1] = [1+4+15] = [20]
        // [3 6]         [1]   [1+6+18]   [25]
        
        // Check results
        REQUIRE(y[0] == Approx(15.0));
        REQUIRE(y[1] == Approx(20.0));
        REQUIRE(y[2] == Approx(25.0));
    }
    
    SECTION("Test dgemv (transpose)") {
        const int m = 3; // Rows
        const int n = 2; // Columns
        double a[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // Column-major: [1 4; 2 5; 3 6]
        double x[3] = {2.0, 3.0, 4.0};
        double y[2] = {1.0, 1.0}; // Initial values
        double alpha = 2.0;
        double beta = 0.5;
        char trans = 'T'; // Transpose
        int lda = 3;
        int incx = 1;
        int incy = 1;

        std::cout << "Testing embedded dgemv (transpose) directly..." << std::endl;
        
        // Call our embedded implementation
        nasoq::embedded::dgemv(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
        
        // Expected result: y = alpha*A'*x + beta*y
        // [1 2 3]   [2]   [0.5]   [2*(2+6+12)+0.5]   [40.5]
        // [4 5 6] * [3] + [0.5] = [2*(8+15+24)+0.5] = [94.5]
        //           [4]
        
        // Check results
        REQUIRE(y[0] == Approx(40.5));
        REQUIRE(y[1] == Approx(94.5));
    }
    
    // Test blocked_2by2_mult function
    SECTION("Test blocked_2by2_mult (1x1 blocks)") {
        int n = 2; // Size of D
        int m = 2; // Number of columns in src/dst
        double D[4] = {2.0, 3.0, 0.0, 0.0}; // Diagonal blocks (D[0]=2.0, D[1]=3.0, no 2x2 blocks)
        double src[4] = {1.0, 4.0, 2.0, 5.0}; // Source matrix (column-major): [1 2; 4 5]
        double dst[4] = {0.0, 0.0, 0.0, 0.0}; // Destination matrix
        int lda = 1; // Leading dimension of src
        int lda_d = 2; // Stride for subdiagonal elements

        std::cout << "Testing embedded blocked_2by2_mult (1x1 blocks) directly..." << std::endl;
        
        // Call our embedded implementation
        nasoq::embedded::blocked_2by2_mult(n, m, D, src, dst, lda, lda_d);
        
        // Expected result:
        // [2 0]   [1 2]   [2*1 2*2]   [2 4]
        // [0 3] * [4 5] = [3*4 3*5] = [12 15]
        
        // Check results
        REQUIRE(dst[0] == Approx(2.0));
        REQUIRE(dst[1] == Approx(12.0));
        REQUIRE(dst[2] == Approx(4.0));
        REQUIRE(dst[3] == Approx(15.0));
    }
    
    SECTION("Test blocked_2by2_mult (2x2 block)") {
        int n = 2; // Size of D
        int m = 2; // Number of columns in src/dst
        double D[3] = {2.0, 3.0, 1.0}; // D[0]=2.0, D[1]=3.0, D[2]=subdiagonal=1.0
        double src[4] = {1.0, 4.0, 2.0, 5.0}; // Source matrix (column-major): [1 2; 4 5]
        double dst[4] = {0.0, 0.0, 0.0, 0.0}; // Destination matrix
        int lda = 1; // Leading dimension of src
        int lda_d = 2; // Stride for subdiagonal elements

        std::cout << "Testing embedded blocked_2by2_mult (2x2 block) directly..." << std::endl;
        
        // Call our embedded implementation
        nasoq::embedded::blocked_2by2_mult(n, m, D, src, dst, lda, lda_d);
        
        // Expected result:
        // [2 1]   [1 2]   [2*1+1*4 2*2+1*5]   [6 9]
        // [1 3] * [4 5] = [1*1+3*4 1*2+3*5] = [13 17]
        
        // Check results
        REQUIRE(dst[0] == Approx(6.0));
        REQUIRE(dst[1] == Approx(13.0));
        REQUIRE(dst[2] == Approx(9.0));
        REQUIRE(dst[3] == Approx(17.0));
    }
    
    // Test dgemm function
    SECTION("Test dgemm (no transpose)") {
        const int m = 2; // Rows of C and A
        const int n = 3; // Columns of C and B
        const int k = 2; // Columns of A, rows of B
        
        // Matrix A (column-major): [1 3; 2 4]
        double a[4] = {1.0, 2.0, 3.0, 4.0};
        
        // Matrix B (column-major): [1 3 5; 2 4 6]
        double b[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        
        // Matrix C (column-major, will be overwritten): [1 4 7; 2 5 8]
        double c[6] = {1.0, 2.0, 4.0, 5.0, 7.0, 8.0};
        
        double alpha = 1.0;
        double beta = 0.0; // C will be completely overwritten
        
        char transa = 'N';
        char transb = 'N';
        
        int lda = 2;
        int ldb = 2;
        int ldc = 2;

        std::cout << "Testing embedded dgemm (no transpose) directly..." << std::endl;
        
        // Call our embedded implementation
        nasoq::embedded::dgemm(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
        
        // Expected result: C = A*B
        // [1 3]   [1 3 5]   [7  15  23]
        // [2 4] * [2 4 6] = [10 22  34]
        
        // Check results
        REQUIRE(c[0] == Approx(7.0));  // c[0,0]
        REQUIRE(c[1] == Approx(10.0)); // c[1,0]
        REQUIRE(c[2] == Approx(15.0)); // c[0,1]
        REQUIRE(c[3] == Approx(22.0)); // c[1,1]
        REQUIRE(c[4] == Approx(23.0)); // c[0,2]
        REQUIRE(c[5] == Approx(34.0)); // c[1,2]
    }
    
    SECTION("Test dgemm (with transpose A)") {
        const int m = 2; // Rows of C and op(A)
        const int n = 2; // Columns of C and op(B)
        const int k = 3; // Columns of op(A), rows of op(B)
        
        // Matrix A (column-major): [1 4; 2 5; 3 6] (3x2)
        // A' = [1 2 3; 4 5 6] (2x3)
        double a[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        
        // Matrix B (column-major): [2 5; 3 6; 4 7] (3x2)
        double b[6] = {2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
        
        // Matrix C (column-major, will be overwritten)
        double c[4] = {1.0, 2.0, 3.0, 4.0};
        
        double alpha = 2.0;
        double beta = 0.5; // C will be scaled and added to
        
        char transa = 'T'; // Transpose A
        char transb = 'N'; // No transpose B
        
        int lda = 3; // Leading dimension of A before transpose
        int ldb = 3; // Leading dimension of B
        int ldc = 2; // Leading dimension of C

        std::cout << "Testing embedded dgemm (with transpose A) directly..." << std::endl;
        
        // Call our embedded implementation
        nasoq::embedded::dgemm(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
        
        // Expected result: C = 2*A'*B + 0.5*C
        // C = 2*[1 2 3; 4 5 6]*[2 5; 3 6; 4 7] + 0.5*[1 3; 2 4]
        // C = 2*[20 41; 47 98] + [0.5 1.5; 1.0 2.0]
        // C = [40.5 83.5; 95.0 198.0]
        
        // Check results
        REQUIRE(c[0] == Approx(40.5));  // c[0,0]
        REQUIRE(c[1] == Approx(95.0));  // c[1,0]
        REQUIRE(c[2] == Approx(83.5));  // c[0,1]
        REQUIRE(c[3] == Approx(198.0)); // c[1,1]
    }
    
    // Test dtrsm function
    SECTION("Test dtrsm (lower triangular)") {
        const int m = 2; // Rows of B
        const int n = 2; // Columns of B
        
        // Lower triangular matrix A (column-major): [2 0; 1 3]
        double a[4] = {2.0, 1.0, 0.0, 3.0};
        
        // Matrix B (column-major, will be overwritten): [6 18; 5 25]
        double b[4] = {6.0, 5.0, 18.0, 25.0};
        
        double alpha = 1.0;
        
        char side = 'L'; // op(A) on left of X
        char uplo = 'L'; // A is lower triangular
        char transa = 'N'; // No transpose
        char diag = 'N'; // Not unit triangular
        
        int lda = 2; // Leading dimension of A
        int ldb = 2; // Leading dimension of B

        std::cout << "Testing embedded dtrsm (lower triangular) directly..." << std::endl;
        
        // Call our embedded implementation
        nasoq::embedded::dtrsm(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
        
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
        REQUIRE(b[0] == Approx(3.0));    // x11
        REQUIRE(b[1] == Approx(2.0/3.0)); // x21
        REQUIRE(b[2] == Approx(9.0));    // x12
        REQUIRE(b[3] == Approx(16.0/3.0)); // x22
    }
    
    SECTION("Test dtrsm (upper triangular)") {
        const int m = 2; // Rows of B
        const int n = 2; // Columns of B
        
        // Upper triangular matrix A (column-major): [2 1; 0 3]
        double a[4] = {2.0, 0.0, 1.0, 3.0};
        
        // Matrix B (column-major, will be overwritten): [12 24; 3 18]
        double b[4] = {12.0, 3.0, 24.0, 18.0};
        
        double alpha = 1.0;
        
        char side = 'L'; // op(A) on left of X
        char uplo = 'U'; // A is upper triangular
        char transa = 'N'; // No transpose
        char diag = 'N'; // Not unit triangular
        
        int lda = 2; // Leading dimension of A
        int ldb = 2; // Leading dimension of B

        std::cout << "Testing embedded dtrsm (upper triangular) directly..." << std::endl;
        
        // Call our embedded implementation
        nasoq::embedded::dtrsm(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
        
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
        REQUIRE(b[0] == Approx(5.5)); // x11
        REQUIRE(b[1] == Approx(1.0)); // x21
        REQUIRE(b[2] == Approx(9.0)); // x12
        REQUIRE(b[3] == Approx(6.0)); // x22
    }
} 