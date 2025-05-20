/**
 * @file embedded_blas.cpp
 * @brief Implementation of embedded-friendly BLAS functions.
 */

#include "nasoq/embedded/embedded_blas.h"
#include <iostream>
#include <cassert>
#include <vector>

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
    std::cout << "*** [EMBEDDED] Using embedded dtrsm implementation (m=" << *m
              << ", n=" << *n << ", side='" << *side << "', uplo='" << *uplo
              << "', transa='" << *transa << "', diag='" << *diag << "') ***" << std::endl;
    std::cout << "**************************************************************\n\n" << std::flush;
    
    // Early return for empty matrices
    if (*m <= 0 || *n <= 0) {
        return;
    }
    
    // If alpha is zero, just set B to zero
    if (*alpha == 0.0) {
        for (int j = 0; j < *n; j++) {
            for (int i = 0; i < *m; i++) {
                b[i + j * (*ldb)] = 0.0;
            }
        }
        return;
    }
    
    // Determine if A is unit diagonal
    bool unit = (*diag == 'U' || *diag == 'u');
    
    // Handle the case where A is on the left side of the equation
    if (*side == 'L' || *side == 'l') {
        // Handle the case where A is not transposed
        if (*transa == 'N' || *transa == 'n') {
            // Handle the case where A is lower triangular
            if (*uplo == 'L' || *uplo == 'l') {
                // Loop through the columns of B
                for (int j = 0; j < *n; j++) {
                    // Scale the current column of B by alpha
                    if (*alpha != 1.0) {
                        for (int i = 0; i < *m; i++) {
                            b[i + j * (*ldb)] *= *alpha;
                        }
                    }
                    
                    // Solve L*x = b for the current column
                    for (int k = 0; k < *m; k++) {
                        if (b[k + j * (*ldb)] != 0.0) {
                            if (!unit) {
                                b[k + j * (*ldb)] /= a[k + k * (*lda)];
                            }
                            
                            for (int i = k + 1; i < *m; i++) {
                                b[i + j * (*ldb)] -= b[k + j * (*ldb)] * a[i + k * (*lda)];
                            }
                        }
                    }
                }
            }
            // Handle the case where A is upper triangular
            else {
                // Loop through the columns of B
                for (int j = 0; j < *n; j++) {
                    // Scale the current column of B by alpha
                    if (*alpha != 1.0) {
                        for (int i = 0; i < *m; i++) {
                            b[i + j * (*ldb)] *= *alpha;
                        }
                    }
                    
                    // Solve U*x = b for the current column
                    for (int k = *m - 1; k >= 0; k--) {
                        if (b[k + j * (*ldb)] != 0.0) {
                            if (!unit) {
                                b[k + j * (*ldb)] /= a[k + k * (*lda)];
                            }
                            
                            for (int i = 0; i < k; i++) {
                                b[i + j * (*ldb)] -= b[k + j * (*ldb)] * a[i + k * (*lda)];
                            }
                        }
                    }
                }
            }
        }
        // Handle the case where A is transposed
        else {
            // Handle the case where A is lower triangular
            if (*uplo == 'L' || *uplo == 'l') {
                // Loop through the columns of B
                for (int j = 0; j < *n; j++) {
                    // Scale the current column of B by alpha
                    if (*alpha != 1.0) {
                        for (int i = 0; i < *m; i++) {
                            b[i + j * (*ldb)] *= *alpha;
                        }
                    }
                    
                    // Solve L'*x = b for the current column
                    for (int i = *m - 1; i >= 0; i--) {
                        double temp = b[i + j * (*ldb)];
                        if (!unit) {
                            temp /= a[i + i * (*lda)];
                        }
                        
                        b[i + j * (*ldb)] = temp;
                        for (int k = 0; k < i; k++) {
                            b[k + j * (*ldb)] -= temp * a[i + k * (*lda)];
                        }
                    }
                }
            }
            // Handle the case where A is upper triangular
            else {
                // Loop through the columns of B
                for (int j = 0; j < *n; j++) {
                    // Scale the current column of B by alpha
                    if (*alpha != 1.0) {
                        for (int i = 0; i < *m; i++) {
                            b[i + j * (*ldb)] *= *alpha;
                        }
                    }
                    
                    // Solve U'*x = b for the current column
                    for (int i = 0; i < *m; i++) {
                        double temp = b[i + j * (*ldb)];
                        if (!unit) {
                            temp /= a[i + i * (*lda)];
                        }
                        
                        b[i + j * (*ldb)] = temp;
                        for (int k = i + 1; k < *m; k++) {
                            b[k + j * (*ldb)] -= temp * a[i + k * (*lda)];
                        }
                    }
                }
            }
        }
    }
    // Handle the case where A is on the right side of the equation
    else {
        // Implementation of right-side triangular solve omitted for brevity
        // This would follow a similar pattern to the left-side case
        std::cout << "[EMBEDDED] Warning: Right-side triangular solve not fully implemented yet." << std::endl;
    }
}

/**
 * @brief Vector dot product (DDOT)
 * 
 * Computes the dot product of two vectors
 */
double dot(int n, const double *a, const double *b) {
    std::cout << "\n\n**************************************************************" << std::endl;
    std::cout << "*** [EMBEDDED] Using embedded dot implementation (n=" << n << ") ***" << std::endl;
    std::cout << "**************************************************************\n\n" << std::flush;
    
    double result = 0.0;
    for (int i = 0; i < n; ++i) {
        result += (a[i] * b[i]);
    }
    return result;
}

/**
 * @brief Vector swap (DSWAP)
 * 
 * Swaps the contents of two vectors
 */
void swap_vector(int n, double *a, double *b, int lda) {
    std::cout << "\n\n**************************************************************" << std::endl;
    std::cout << "*** [EMBEDDED] Using embedded swap_vector implementation (n=" << n << ", lda=" << lda << ") ***" << std::endl;
    std::cout << "**************************************************************\n\n" << std::flush;
    
    double tmp = 0;
    if (lda == 1) {
        // Optimize for unit stride
        for (int i = 0; i < n; ++i) {
            tmp = *(a + i);
            *(a + i) = *(b + i);
            *(b + i) = tmp;
        }
    } else {
        // General case for non-unit stride
        for (int i = 0; i < n; ++i) {
            tmp = *(a + i * lda);
            *(a + i * lda) = *(b + i * lda);
            *(b + i * lda) = tmp;
        }
    }
}

void sym_sytrf(double *A, int n, const int stride, int *nbpivot, double critere) {
    std::cout << "\n\n**************************************************************" << std::endl;
    std::cout << "*** [EMBEDDED] Using embedded sym_sytrf implementation (n=" << n << ", stride=" << stride << ") ***" << std::endl;
    std::cout << "**************************************************************\n\n" << std::flush;
    
    const int iun = 1;
    const double one = 1.0;
    double *tmp, *tmp1;
    
    for (int k = 0; k < n; k++) {
        tmp = A + k * (stride + 1);
        
        #if 0 // Small pivot detection (disabled by default, same as in original implementation)
        if (std::abs(*tmp) <= critere) {
            (*tmp) = critere;
            (*nbpivot)++;
        }
        #endif
        
        tmp1 = tmp + 1;
        int tmp_dim = n - k - 1;
        double sca_tmp = one / (*tmp);
        
        // Scale column k below the diagonal by 1/A[k,k]
        dscal(&tmp_dim, &sca_tmp, tmp1, &iun);
        
        // Perform symmetric rank-1 update of the trailing submatrix
        int dimx = n - k - 1;
        double diag = -(*tmp);
        double *tmp1_stride = tmp1 + stride;
        
        char uplo = 'L';
        dsyr(&uplo, &dimx, &diag, tmp1, &iun, tmp1_stride, &stride);
    }
}

// Helper functions for dlapmt
template<typename T>
inline const T& embedded_minimum(const T& a, const T& b) {
    return a <= b ? a : b;
}

template<typename T>
inline const T& embedded_maximum(const T& a, const T& b) {
    return a >= b ? a : b;
}

/**
 * @brief Permute columns of a matrix (DLAPMT)
 * 
 * Embedded implementation that doesn't depend on external LAPACK.
 */
int dlapmt(int matrix_layout, int forwrd, int m, int n, 
           double *x, int ldx, int *k) {
    
    std::cout << "\n\n**************************************************************" << std::endl;
    std::cout << "*** [EMBEDDED] Using embedded dlapmt implementation (m=" << m 
              << ", n=" << n << ", forwrd=" << (forwrd ? "true" : "false") << ") ***" << std::endl;
    std::cout << "**************************************************************\n\n" << std::flush;
    
    // Constants for matrix layout
    const int LAPACK_ROW_MAJOR = 101;
    const int LAPACK_COL_MAJOR = 102;
    
    // Function to transpose matrix blocks
    auto transpose_into = [](double* out_x_t, int ldx_t, const double* x, int ldx, 
                             int m, int n, int matrix_layout) {
        int i_max, j_max;
        if (LAPACK_COL_MAJOR == matrix_layout) {
            j_max = n;
            i_max = m;
        } else {
            j_max = m;
            i_max = n;
        }
        j_max = embedded_minimum(j_max, ldx_t);
        i_max = embedded_minimum(i_max, ldx);

        for (int i = 0; i < i_max; ++i) {
            for (int j = 0; j < j_max; ++j) {
                out_x_t[i*ldx_t + j] = x[j*ldx + i];
            }
        }
    };
    
    // Handle row major layout by transposing to column major, permuting, then transposing back
    if (LAPACK_ROW_MAJOR == matrix_layout) {
        int ldx_t = embedded_maximum(1, m);
        if (ldx < n) return -6;  // Invalid ldx parameter
        
        // Create temporary buffer for transposed matrix
        std::vector<double> x_t(ldx_t * embedded_maximum(1, n));
        
        // Transpose input matrix to column major format
        transpose_into(x_t.data(), ldx_t, x, ldx, m, n, matrix_layout);
        
        // Apply column permutation on transposed matrix in column major format
        int info = dlapmt(LAPACK_COL_MAJOR, forwrd, m, n, x_t.data(), ldx_t, k);
        if (info < 0) return info;
        
        // Transpose result back to row major format
        transpose_into(x, ldx, x_t.data(), ldx_t, m, n, LAPACK_COL_MAJOR);
        
        return 0;
    } 
    // For column major layout, apply permutation directly
    else if (LAPACK_COL_MAJOR == matrix_layout) {
        if (ldx < m) return -6;  // Invalid ldx parameter
        
        // Create temporary space for column swap operations
        std::vector<double> temp_col(m);
        std::vector<int> perm(n);
        
        // Initialize permutation tracking array
        for (int i = 0; i < n; i++) {
            perm[i] = i;
        }
        
        if (forwrd) {
            // Forward permutation: X(*,K(J)) is moved to X(*,J)
            for (int j = 0; j < n; j++) {
                // Skip if column is already in correct position
                if (perm[j] == j) continue;
                
                int curr_col = j;
                int dest_col = k[j] - 1;  // Convert from 1-indexed to 0-indexed
                
                // Save the current column
                for (int i = 0; i < m; i++) {
                    temp_col[i] = x[i + curr_col * ldx];
                }
                
                // Move columns in a cyclic fashion until we return to start
                while (dest_col != j) {
                    // Move destination column to current position
                    for (int i = 0; i < m; i++) {
                        x[i + curr_col * ldx] = x[i + dest_col * ldx];
                    }
                    
                    // Mark this permutation as done
                    perm[curr_col] = perm[dest_col];
                    
                    // Move to next column in cycle
                    curr_col = dest_col;
                    dest_col = k[curr_col] - 1;  // Convert from 1-indexed to 0-indexed
                }
                
                // Place saved column in final position
                for (int i = 0; i < m; i++) {
                    x[i + curr_col * ldx] = temp_col[i];
                }
                
                // Mark final permutation as done
                perm[curr_col] = j;
            }
        } else {
            // Backward permutation: X(*,J) is moved to X(*,K(J))
            for (int j = 0; j < n; j++) {
                // Skip if column is already in correct position
                if (perm[j] == j) continue;
                
                int curr_col = j;
                int dest_col = k[j] - 1;  // Convert from 1-indexed to 0-indexed
                
                // Save the current column
                for (int i = 0; i < m; i++) {
                    temp_col[i] = x[i + curr_col * ldx];
                }
                
                // Move columns in a cyclic fashion until we return to start
                while (dest_col != j) {
                    // Move destination column to current position
                    for (int i = 0; i < m; i++) {
                        x[i + curr_col * ldx] = x[i + dest_col * ldx];
                    }
                    
                    // Mark this permutation as done
                    perm[curr_col] = perm[dest_col];
                    
                    // Move to next column in cycle
                    curr_col = dest_col;
                    dest_col = k[curr_col] - 1;  // Convert from 1-indexed to 0-indexed
                }
                
                // Place saved column in final position
                for (int i = 0; i < m; i++) {
                    x[i + curr_col * ldx] = temp_col[i];
                }
                
                // Mark final permutation as done
                perm[curr_col] = j;
            }
        }
        
        return 0;
    } else {
        return -1;  // Invalid matrix_layout parameter
    }
}

} // namespace embedded
} // namespace nasoq 