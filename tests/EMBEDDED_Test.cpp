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
} 