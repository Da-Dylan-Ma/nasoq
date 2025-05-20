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

void dgemv(const char *trans, const int *m, const int *n,
           const double *alpha, const double *a, const int *lda,
           const double *x, const int *incx,
           const double *beta, double *y, const int *incy) {
    std::cout << "\n\n**************************************************************" << std::endl;
    std::cout << "*** [EMBEDDED] Using embedded dgemv implementation (m=" << *m << ", n=" << *n 
              << ", trans='" << *trans << "', alpha=" << *alpha << ", beta=" << *beta << ") ***" << std::endl;
    std::cout << "**************************************************************\n\n" << std::flush;
    
    // Check for special cases where we can return early
    if (*m <= 0 || *n <= 0) {
        return;
    }
    
    if (*alpha == 0.0 && *beta == 1.0) {
        return; // No operation needed: y = y
    }
    
    // Determine operation type (normal or transpose)
    bool do_trans = (*trans == 'T' || *trans == 't' || *trans == 'C' || *trans == 'c');
    
    // If beta != 1.0, scale the output vector
    if (*beta != 1.0) {
        int len_y = do_trans ? *m : *n;
        if (*beta == 0.0) {
            // Special case: beta = 0, just zero out y
            if (*incy == 1) {
                for (int i = 0; i < len_y; i++) {
                    y[i] = 0.0;
                }
            } else {
                int iy = 0;
                for (int i = 0; i < len_y; i++) {
                    y[iy] = 0.0;
                    iy += *incy;
                }
            }
        } else {
            // General case: scale y by beta
            if (*incy == 1) {
                for (int i = 0; i < len_y; i++) {
                    y[i] *= *beta;
                }
            } else {
                int iy = 0;
                for (int i = 0; i < len_y; i++) {
                    y[iy] *= *beta;
                    iy += *incy;
                }
            }
        }
    }
    
    // If alpha == 0, we can return after scaling y
    if (*alpha == 0.0) {
        return;
    }
    
    // Handle non-transposed operation
    if (!do_trans) {
        // y = alpha*A*x + beta*y
        if (*incx == 1 && *incy == 1) {
            // Optimized case for unit strides
            for (int j = 0; j < *n; j++) {
                double x_val = *alpha * x[j];
                for (int i = 0; i < *m; i++) {
                    y[i] += a[i + j * (*lda)] * x_val;
                }
            }
        } else {
            // General case for non-unit strides
            int ix = (*incx > 0) ? 0 : ((-*n + 1) * (*incx));
            
            for (int j = 0; j < *n; j++) {
                double x_val = *alpha * x[ix];
                int iy = (*incy > 0) ? 0 : ((-*m + 1) * (*incy));
                
                for (int i = 0; i < *m; i++) {
                    y[iy] += a[i + j * (*lda)] * x_val;
                    iy += *incy;
                }
                
                ix += *incx;
            }
        }
    } else {
        // y = alpha*A'*x + beta*y
        int iy = (*incy > 0) ? 0 : ((-*n + 1) * (*incy));
        
        if (*incx == 1) {
            // Optimized case when incx = 1
            for (int j = 0; j < *n; j++) {
                double temp = 0.0;
                for (int i = 0; i < *m; i++) {
                    temp += a[i + j * (*lda)] * x[i];
                }
                y[iy] += *alpha * temp;
                iy += *incy;
            }
        } else {
            // General case for non-unit stride in x
            for (int j = 0; j < *n; j++) {
                double temp = 0.0;
                int ix = (*incx > 0) ? 0 : ((-*m + 1) * (*incx));
                
                for (int i = 0; i < *m; i++) {
                    temp += a[i + j * (*lda)] * x[ix];
                    ix += *incx;
                }
                
                y[iy] += *alpha * temp;
                iy += *incy;
            }
        }
    }
}

void blocked_2by2_mult(int n, int m, double *D, double *src, double *dst, int lda, int lda_d) {
    std::cout << "\n\n**************************************************************" << std::endl;
    std::cout << "*** [EMBEDDED] Using embedded blocked_2by2_mult implementation (n=" << n << ", m=" << m << ") ***" << std::endl;
    std::cout << "**************************************************************\n\n" << std::flush;
    
    // Loop through each block in the diagonal
    for (int i = 0; i < n;) {
        if (D[i + lda_d] == 0) { 
            // 1x1 block - simple scaling of the row
            double d_val = D[i];
            
            for (int j = 0; j < m; ++j) {
                dst[i * m + j] = d_val * src[i * lda + j];
            }
            
            // Move to the next block
            i++;
        } else {
            // 2x2 block - need to handle cross-terms
            double d1 = D[i];         // D[i,i]
            double d2 = D[i + 1];     // D[i+1,i+1]
            double off_d = D[i + lda_d]; // D[i,i+1] = D[i+1,i]
            
            for (int j = 0; j < m; ++j) {
                // Compute output for the first row
                dst[i * m + j] = d1 * src[i * lda + j] + off_d * src[(i + 1) * lda + j];
                
                // Compute output for the second row
                dst[(i + 1) * m + j] = off_d * src[i * lda + j] + d2 * src[(i + 1) * lda + j];
            }
            
            // Skip the next row since it was part of the 2x2 block
            i += 2;
        }
    }
}

} // namespace embedded
} // namespace nasoq 