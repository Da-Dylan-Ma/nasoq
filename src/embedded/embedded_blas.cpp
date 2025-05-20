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

void dgemm(const char *transa, const char *transb, const int *m, const int *n, const int *k,
           const double *alpha, const double *a, const int *lda, const double *b, const int *ldb,
           const double *beta, double *c, const int *ldc) {
    std::cout << "\n\n**************************************************************" << std::endl;
    std::cout << "*** [EMBEDDED] Using embedded dgemm implementation (m=" << *m << ", n=" << *n 
              << ", k=" << *k << ", transa='" << *transa << "', transb='" << *transb 
              << "', alpha=" << *alpha << ", beta=" << *beta << ") ***" << std::endl;
    std::cout << "**************************************************************\n\n" << std::flush;
    
    // Check for special cases where we can return early
    if (*m <= 0 || *n <= 0 || *k <= 0) {
        return;
    }
    
    if (*alpha == 0.0 && *beta == 1.0) {
        return; // No operation needed: C = C
    }
    
    // Determine operation types based on transA and transB flags
    bool trans_a = (*transa == 'T' || *transa == 't' || *transa == 'C' || *transa == 'c');
    bool trans_b = (*transb == 'T' || *transb == 't' || *transb == 'C' || *transb == 'c');
    
    // Scale C by beta
    if (*beta != 1.0) {
        if (*beta == 0.0) {
            // Special case: beta = 0, just zero out C
            for (int j = 0; j < *n; j++) {
                for (int i = 0; i < *m; i++) {
                    c[i + j * (*ldc)] = 0.0;
                }
            }
        } else {
            // General case: scale C by beta
            for (int j = 0; j < *n; j++) {
                for (int i = 0; i < *m; i++) {
                    c[i + j * (*ldc)] *= *beta;
                }
            }
        }
    }
    
    // If alpha == 0, we can return after scaling C by beta
    if (*alpha == 0.0) {
        return;
    }
    
    // Handle all four cases based on transA and transB
    if (!trans_a && !trans_b) {
        // C = alpha*A*B + beta*C
        for (int j = 0; j < *n; j++) {
            for (int l = 0; l < *k; l++) {
                double temp = *alpha * b[l + j * (*ldb)];
                for (int i = 0; i < *m; i++) {
                    c[i + j * (*ldc)] += a[i + l * (*lda)] * temp;
                }
            }
        }
    } else if (trans_a && !trans_b) {
        // C = alpha*A'*B + beta*C
        for (int j = 0; j < *n; j++) {
            for (int i = 0; i < *m; i++) {
                double temp = 0.0;
                for (int l = 0; l < *k; l++) {
                    temp += a[l + i * (*lda)] * b[l + j * (*ldb)];
                }
                c[i + j * (*ldc)] += *alpha * temp;
            }
        }
    } else if (!trans_a && trans_b) {
        // C = alpha*A*B' + beta*C
        for (int j = 0; j < *n; j++) {
            for (int i = 0; i < *m; i++) {
                double temp = 0.0;
                for (int l = 0; l < *k; l++) {
                    temp += a[i + l * (*lda)] * b[j + l * (*ldb)];
                }
                c[i + j * (*ldc)] += *alpha * temp;
            }
        }
    } else { // trans_a && trans_b
        // C = alpha*A'*B' + beta*C
        for (int j = 0; j < *n; j++) {
            for (int i = 0; i < *m; i++) {
                double temp = 0.0;
                for (int l = 0; l < *k; l++) {
                    temp += a[l + i * (*lda)] * b[j + l * (*ldb)];
                }
                c[i + j * (*ldc)] += *alpha * temp;
            }
        }
    }
}

void dtrsm(const char *side, const char *uplo, const char *transa, const char *diag,
           const int *m, const int *n, const double *alpha,
           const double *a, const int *lda, double *b, const int *ldb) {
    std::cout << "\n\n**************************************************************" << std::endl;
    std::cout << "*** [EMBEDDED] Using embedded dtrsm implementation (m=" << *m << ", n=" << *n 
              << ", side='" << *side << "', uplo='" << *uplo << "', transa='" << *transa 
              << "', diag='" << *diag << "', alpha=" << *alpha << ") ***" << std::endl;
    std::cout << "**************************************************************\n\n" << std::flush;
    
    // Check for special cases
    if (*m <= 0 || *n <= 0) {
        return;
    }
    
    if (*alpha == 0.0) {
        // If alpha is zero, set B to zero and return
        for (int j = 0; j < *n; j++) {
            for (int i = 0; i < *m; i++) {
                b[i + j * (*ldb)] = 0.0;
            }
        }
        return;
    }
    
    // Determine operation parameters
    bool left_side = (*side == 'L' || *side == 'l');
    bool upper = (*uplo == 'U' || *uplo == 'u');
    bool trans_a = (*transa == 'T' || *transa == 't' || *transa == 'C' || *transa == 'c');
    bool unit_diag = (*diag == 'U' || *diag == 'u');
    
    // Scale B by alpha if alpha != 1.0
    if (*alpha != 1.0) {
        for (int j = 0; j < *n; j++) {
            for (int i = 0; i < *m; i++) {
                b[i + j * (*ldb)] *= *alpha;
            }
        }
    }
    
    // Handle the different operation cases
    if (left_side) {
        // op(A) * X = B, solve for X (overwriting B)
        if (!trans_a) {
            // A * X = B
            if (upper) {
                // A is upper triangular
                for (int j = 0; j < *n; j++) {
                    // Process each column of B
                    for (int k = *m - 1; k >= 0; k--) {
                        // Skip division by zero diagonal elements
                        if (!unit_diag && a[k + k * (*lda)] == 0.0) {
                            // Division by zero - not handling here, could set to inf/nan or error
                            continue;
                        }
                        
                        // Divide by diagonal element (unless unit diagonal)
                        if (!unit_diag) {
                            b[k + j * (*ldb)] /= a[k + k * (*lda)];
                        }
                        
                        // Update other elements in this column
                        double temp = b[k + j * (*ldb)];
                        for (int i = 0; i < k; i++) {
                            b[i + j * (*ldb)] -= temp * a[i + k * (*lda)];
                        }
                    }
                }
            } else {
                // A is lower triangular
                for (int j = 0; j < *n; j++) {
                    // Process each column of B
                    for (int k = 0; k < *m; k++) {
                        // Skip division by zero diagonal elements
                        if (!unit_diag && a[k + k * (*lda)] == 0.0) {
                            // Division by zero - not handling here, could set to inf/nan or error
                            continue;
                        }
                        
                        // Divide by diagonal element (unless unit diagonal)
                        if (!unit_diag) {
                            b[k + j * (*ldb)] /= a[k + k * (*lda)];
                        }
                        
                        // Update remaining elements in this column
                        double temp = b[k + j * (*ldb)];
                        for (int i = k + 1; i < *m; i++) {
                            b[i + j * (*ldb)] -= temp * a[i + k * (*lda)];
                        }
                    }
                }
            }
        } else {
            // A' * X = B
            if (upper) {
                // A is upper triangular, so A' is lower triangular
                for (int j = 0; j < *n; j++) {
                    // Process each column of B
                    for (int k = 0; k < *m; k++) {
                        // First compute the dot product of preceding rows of A' with X
                        double temp = b[k + j * (*ldb)];
                        for (int i = 0; i < k; i++) {
                            temp -= a[i + k * (*lda)] * b[i + j * (*ldb)];
                        }
                        
                        // Then divide by diagonal (unless unit diagonal)
                        if (!unit_diag) {
                            if (a[k + k * (*lda)] == 0.0) {
                                // Division by zero - not handling here
                                continue;
                            }
                            temp /= a[k + k * (*lda)];
                        }
                        
                        b[k + j * (*ldb)] = temp;
                    }
                }
            } else {
                // A is lower triangular, so A' is upper triangular
                for (int j = 0; j < *n; j++) {
                    // Process each column of B
                    for (int k = *m - 1; k >= 0; k--) {
                        // First compute the dot product of preceding rows of A' with X
                        double temp = b[k + j * (*ldb)];
                        for (int i = k + 1; i < *m; i++) {
                            temp -= a[i + k * (*lda)] * b[i + j * (*ldb)];
                        }
                        
                        // Then divide by diagonal (unless unit diagonal)
                        if (!unit_diag) {
                            if (a[k + k * (*lda)] == 0.0) {
                                // Division by zero - not handling here
                                continue;
                            }
                            temp /= a[k + k * (*lda)];
                        }
                        
                        b[k + j * (*ldb)] = temp;
                    }
                }
            }
        }
    } else {
        // X * op(A) = B, solve for X (overwriting B)
        if (!trans_a) {
            // X * A = B
            if (upper) {
                // A is upper triangular
                for (int j = 0; j < *n; j++) {
                    // Skip division by zero diagonal elements
                    if (!unit_diag && a[j + j * (*lda)] == 0.0) {
                        // Division by zero - not handling here
                        continue;
                    }
                    
                    // Divide by diagonal element (unless unit diagonal)
                    if (!unit_diag) {
                        for (int i = 0; i < *m; i++) {
                            b[i + j * (*ldb)] /= a[j + j * (*lda)];
                        }
                    }
                    
                    // Update other columns
                    for (int k = j + 1; k < *n; k++) {
                        double temp = a[j + k * (*lda)];
                        for (int i = 0; i < *m; i++) {
                            b[i + k * (*ldb)] -= b[i + j * (*ldb)] * temp;
                        }
                    }
                }
            } else {
                // A is lower triangular
                for (int j = *n - 1; j >= 0; j--) {
                    // Skip division by zero diagonal elements
                    if (!unit_diag && a[j + j * (*lda)] == 0.0) {
                        // Division by zero - not handling here
                        continue;
                    }
                    
                    // Divide by diagonal element (unless unit diagonal)
                    if (!unit_diag) {
                        for (int i = 0; i < *m; i++) {
                            b[i + j * (*ldb)] /= a[j + j * (*lda)];
                        }
                    }
                    
                    // Update other columns
                    for (int k = 0; k < j; k++) {
                        double temp = a[j + k * (*lda)];
                        for (int i = 0; i < *m; i++) {
                            b[i + k * (*ldb)] -= b[i + j * (*ldb)] * temp;
                        }
                    }
                }
            }
        } else {
            // X * A' = B
            if (upper) {
                // A is upper triangular, so A' is lower triangular
                for (int j = *n - 1; j >= 0; j--) {
                    // First compute the effect of preceding columns
                    for (int i = 0; i < *m; i++) {
                        double temp = b[i + j * (*ldb)];
                        for (int k = 0; k < j; k++) {
                            temp -= b[i + k * (*ldb)] * a[k + j * (*lda)];
                        }
                        
                        // Then divide by diagonal (unless unit diagonal)
                        if (!unit_diag) {
                            if (a[j + j * (*lda)] == 0.0) {
                                // Division by zero - not handling here
                                continue;
                            }
                            temp /= a[j + j * (*lda)];
                        }
                        
                        b[i + j * (*ldb)] = temp;
                    }
                }
            } else {
                // A is lower triangular, so A' is upper triangular
                for (int j = 0; j < *n; j++) {
                    // First compute the effect of preceding columns
                    for (int i = 0; i < *m; i++) {
                        double temp = b[i + j * (*ldb)];
                        for (int k = j + 1; k < *n; k++) {
                            temp -= b[i + k * (*ldb)] * a[k + j * (*lda)];
                        }
                        
                        // Then divide by diagonal (unless unit diagonal)
                        if (!unit_diag) {
                            if (a[j + j * (*lda)] == 0.0) {
                                // Division by zero - not handling here
                                continue;
                            }
                            temp /= a[j + j * (*lda)];
                        }
                        
                        b[i + j * (*ldb)] = temp;
                    }
                }
            }
        }
    }
}

} // namespace embedded
} // namespace nasoq 