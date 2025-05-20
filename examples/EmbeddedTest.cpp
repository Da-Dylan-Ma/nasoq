#include <iostream>
#include "nasoq/embedded/embedded_blas.h"

int main() {
    std::cout << "Starting embedded BLAS functions test..." << std::endl;
    
    // Test dscal
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
    
    // Test dsyr
    const int m = 3;
    double a[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}; // 3x3 identity matrix
    double y[3] = {1.0, 2.0, 3.0};
    const double beta = -1.0; // Negative to make the update more visible
    const char uplo = 'L';
    int incy = 1;
    int lda = 3;
    
    std::cout << "Original matrix a:" << std::endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            std::cout << a[i + j*lda] << " ";
        }
        std::cout << std::endl;
    }
    
    // Call embedded dsyr
    nasoq::embedded::dsyr(&uplo, &m, &beta, y, &incy, a, &lda);
    
    std::cout << "After dsyr (rank-1 update with y and beta=-1.0):" << std::endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            std::cout << a[i + j*lda] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "Embedded BLAS functions test completed successfully!" << std::endl;
    return 0;
} 