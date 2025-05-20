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
} 