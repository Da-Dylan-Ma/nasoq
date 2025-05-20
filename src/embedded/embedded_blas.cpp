/**
 * @file embedded_blas.cpp
 * @brief Implementation of embedded-friendly BLAS functions.
 */

#include "nasoq/embedded/embedded_blas.h"
#include <iostream>
#include <cassert>

namespace nasoq {
namespace embedded {

void dscal(const int *n, const double *alpha, double *x, const int *incx) {
    std::cout << "\n\n**************************************************************" << std::endl;
    std::cout << "*** [EMBEDDED] Using embedded dscal implementation (n=" << *n << ", alpha=" << *alpha << ") ***" << std::endl;
    std::cout << "**************************************************************\n\n" << std::flush;
    
    // Check for special cases
    if (*n <= 0 || *incx <= 0) {
        return;
    }

    // Handle stride = 1 case efficiently
    if (*incx == 1) {
        for (int i = 0; i < *n; i++) {
            x[i] *= *alpha;
        }
    } else {
        // Handle general stride case
        int ix = 0;
        for (int i = 0; i < *n; i++) {
            x[ix] *= *alpha;
            ix += *incx;
        }
    }
}

void dsyr(const char *uplo, const int *n, const double *alpha,
          const double *x, const int *incx, double *a, const int *lda) {
    std::cout << "\n\n**************************************************************" << std::endl;
    std::cout << "*** [EMBEDDED] Using embedded dsyr implementation (n=" << *n << ", alpha=" << *alpha << ", uplo=" << *uplo << ") ***" << std::endl; 
    std::cout << "**************************************************************\n\n" << std::flush;
    
    // Check for special cases
    if (*n <= 0 || *alpha == 0.0) {
        return;
    }

    // Determine whether we're updating the lower ('L') or upper ('U') triangular part
    bool lower = (*uplo == 'L' || *uplo == 'l');

    // Calculate the stride for x vector
    int ix = (*incx < 0) ? ((-(*n) + 1) * (*incx)) : 0;

    // Perform the symmetric rank-1 update
    if (*incx == 1) {
        // Efficient implementation for unit stride
        for (int j = 0; j < *n; j++) {
            // Skip computation if x[j] is zero
            if (x[j] != 0.0) {
                double temp = *alpha * x[j];
                
                if (lower) {
                    // Update lower triangular part including diagonal
                    for (int i = j; i < *n; i++) {
                        a[i + j * (*lda)] += x[i] * temp;
                    }
                } else {
                    // Update upper triangular part including diagonal
                    for (int i = 0; i <= j; i++) {
                        a[i + j * (*lda)] += x[i] * temp;
                    }
                }
            }
        }
    } else {
        // Implementation for non-unit stride
        for (int j = 0; j < *n; j++) {
            int jx = ix + j * (*incx);
            if (x[jx / (*incx)] != 0.0) {
                double temp = *alpha * x[jx / (*incx)];
                
                if (lower) {
                    // Update lower triangular part including diagonal
                    int kx = jx;
                    for (int i = j; i < *n; i++) {
                        a[i + j * (*lda)] += x[kx / (*incx)] * temp;
                        kx += *incx;
                    }
                } else {
                    // Update upper triangular part including diagonal
                    int kx = ix;
                    for (int i = 0; i <= j; i++) {
                        a[i + j * (*lda)] += x[kx / (*incx)] * temp;
                        kx += *incx;
                    }
                }
            }
        }
    }
}

void dcopy(const int *n, const double *x, const int *incx, 
           double *y, const int *incy) {
    std::cout << "\n\n**************************************************************" << std::endl;
    std::cout << "*** [EMBEDDED] Using embedded dcopy implementation (n=" << *n << ") ***" << std::endl;
    std::cout << "**************************************************************\n\n" << std::flush;
    
    // Check for special cases
    if (*n <= 0) {
        return;
    }
    
    // Handle different stride cases
    if (*incx == 1 && *incy == 1) {
        // Optimized case: unit stride for both vectors
        for (int i = 0; i < *n; i++) {
            y[i] = x[i];
        }
    } else {
        // General case: non-unit stride
        int ix = (*incx > 0) ? 0 : ((-*n + 1) * (*incx));
        int iy = (*incy > 0) ? 0 : ((-*n + 1) * (*incy));
        
        for (int i = 0; i < *n; i++) {
            y[iy] = x[ix];
            ix += *incx;
            iy += *incy;
        }
    }
}

void blocked_2by2_solver(int n, double *D, double *rhs, int n_rhs, int lda, int lda_d) {
    std::cout << "\n\n**************************************************************" << std::endl;
    std::cout << "*** [EMBEDDED] Using embedded blocked_2by2_solver implementation (n=" << n << ", n_rhs=" << n_rhs << ") ***" << std::endl;
    std::cout << "**************************************************************\n\n" << std::flush;
    
    // Loop through the diagonal blocks
    for (int i = 0; i < n; ++i) {
        if (D[i + lda_d] == 0) { 
            // Simple 1x1 block - just scale by the inverse of the diagonal element
            assert(D[i] != 0); // Ensure diagonal is non-zero
            double tmp = 1.0 / D[i];
            
            // Scale the corresponding row of the right-hand side
            for (int j = 0; j < n_rhs; ++j) {
                rhs[i * lda + j] *= tmp;
            }
        } else {
            // 2x2 block - solve using Cramer's rule
            double subdiag = D[i + lda_d];
            double determinant = D[i] * D[i + 1] - subdiag * subdiag;
            double one_over_det = 1.0 / determinant;
            
            // Solve each right-hand side
            for (int j = 0; j < n_rhs; ++j) {
                double x1 = rhs[i * lda + j];
                double x2 = rhs[(i + 1) * lda + j];
                
                // Compute the solution
                rhs[i * lda + j] = (x1 * D[i + 1] - x2 * subdiag) * one_over_det;
                rhs[(i + 1) * lda + j] = (x2 * D[i] - x1 * subdiag) * one_over_det;
            }
            
            // Skip the next row since it was part of the 2x2 block
            i++;
        }
    }
}

} // namespace embedded
} // namespace nasoq 