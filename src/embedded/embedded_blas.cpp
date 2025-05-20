/**
 * @file embedded_blas.cpp
 * @brief Implementation of embedded-friendly BLAS functions.
 */

#include "nasoq/embedded/embedded_blas.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

namespace nasoq {
namespace embedded {

void dscal(const int *n, const double *alpha, double *x, const int *incx) {
    std::cout << "\n\n**************************************************************" << std::endl;
    std::cout << "*** [EMBEDDED] Using embedded dscal implementation (n=" << *n << ", alpha=" << *alpha << ") ***" << std::endl;
    std::cout << "**************************************************************\n\n" << std::flush;
    
    if (*n <= 0 || *incx <= 0) {
        return;
    }

    if (*incx == 1) {
        for (int i = 0; i < *n; i++) {
            x[i] *= *alpha;
        }
    } else {
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
    
    if (*n <= 0 || *alpha == 0.0) {
        return;
    }

    bool lower = (*uplo == 'L' || *uplo == 'l');

    int ix = (*incx < 0) ? ((-(*n) + 1) * (*incx)) : 0;

    if (*incx == 1) {
        for (int j = 0; j < *n; j++) {
            if (x[j] != 0.0) {
                double temp = *alpha * x[j];
                
                if (lower) {
                    for (int i = j; i < *n; i++) {
                        a[i + j * (*lda)] += x[i] * temp;
                    }
                } else {
                    for (int i = 0; i <= j; i++) {
                        a[i + j * (*lda)] += x[i] * temp;
                    }
                }
            }
        }
    } else {
        for (int j = 0; j < *n; j++) {
            int jx = ix + j * (*incx);
            if (x[jx / (*incx)] != 0.0) {
                double temp = *alpha * x[jx / (*incx)];
                
                if (lower) {
                    int kx = jx;
                    for (int i = j; i < *n; i++) {
                        a[i + j * (*lda)] += x[kx / (*incx)] * temp;
                        kx += *incx;
                    }
                } else {
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
    
    if (*n <= 0) {
        return;
    }
    
    if (*incx == 1 && *incy == 1) {
        for (int i = 0; i < *n; i++) {
            y[i] = x[i];
        }
    } else {
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
    
    for (int i = 0; i < n; ++i) {
        if (D[i + lda_d] == 0) {
            assert(D[i] != 0);
            double tmp = 1.0 / D[i];
            
            for (int j = 0; j < n_rhs; ++j) {
                rhs[i * lda + j] *= tmp;
            }
        } else {
            double subdiag = D[i + lda_d];
            double determinant = D[i] * D[i + 1] - subdiag * subdiag;
            double one_over_det = 1.0 / determinant;
            
            for (int j = 0; j < n_rhs; ++j) {
                double x1 = rhs[i * lda + j];
                double x2 = rhs[(i + 1) * lda + j];
                
                rhs[i * lda + j] = (x1 * D[i + 1] - x2 * subdiag) * one_over_det;
                rhs[(i + 1) * lda + j] = (x2 * D[i] - x1 * subdiag) * one_over_det;
            }
            
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
    
    if (*m <= 0 || *n <= 0) {
        return;
    }
    
    if (*alpha == 0.0 && *beta == 1.0) {
        return;
    }
    
    bool do_trans = (*trans == 'T' || *trans == 't' || *trans == 'C' || *trans == 'c');
    
    if (*beta != 1.0) {
        int len_y = do_trans ? *m : *n;
        if (*beta == 0.0) {
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
    
    if (*alpha == 0.0) {
        return;
    }
    
    if (!do_trans) {
        if (*incx == 1 && *incy == 1) {
            for (int j = 0; j < *n; j++) {
                double x_val = *alpha * x[j];
                for (int i = 0; i < *m; i++) {
                    y[i] += a[i + j * (*lda)] * x_val;
                }
            }
        } else {
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
        int iy = (*incy > 0) ? 0 : ((-*n + 1) * (*incy));
        
        if (*incx == 1) {
            for (int j = 0; j < *n; j++) {
                double temp = 0.0;
                for (int i = 0; i < *m; i++) {
                    temp += a[i + j * (*lda)] * x[i];
                }
                y[iy] += *alpha * temp;
                iy += *incy;
            }
        } else {
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
    
    for (int i = 0; i < n;) {
        if (D[i + lda_d] == 0) {
            double d_val = D[i];
            
            for (int j = 0; j < m; ++j) {
                dst[i * m + j] = d_val * src[i * lda + j];
            }
            
            i++;
        } else {
            double d1 = D[i];
            double d2 = D[i + 1];
            double off_d = D[i + lda_d];
            
            for (int j = 0; j < m; ++j) {
                dst[i * m + j] = d1 * src[i * lda + j] + off_d * src[(i + 1) * lda + j];
                
                dst[(i + 1) * m + j] = off_d * src[i * lda + j] + d2 * src[(i + 1) * lda + j];
            }
            
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
    
    if (*m <= 0 || *n <= 0 || *k <= 0) {
        return;
    }
    
    if (*alpha == 0.0 && *beta == 1.0) {
        return;
    }
    
    bool trans_a = (*transa == 'T' || *transa == 't' || *transa == 'C' || *transa == 'c');
    bool trans_b = (*transb == 'T' || *transb == 't' || *transb == 'C' || *transb == 'c');
    
    if (*beta != 1.0) {
        if (*beta == 0.0) {
            for (int j = 0; j < *n; j++) {
                for (int i = 0; i < *m; i++) {
                    c[i + j * (*ldc)] = 0.0;
                }
            }
        } else {
            for (int j = 0; j < *n; j++) {
                for (int i = 0; i < *m; i++) {
                    c[i + j * (*ldc)] *= *beta;
                }
            }
        }
    }
    
    if (*alpha == 0.0) {
        return;
    }
    
    if (!trans_a && !trans_b) {
        for (int j = 0; j < *n; j++) {
            for (int l = 0; l < *k; l++) {
                double temp = *alpha * b[l + j * (*ldb)];
                for (int i = 0; i < *m; i++) {
                    c[i + j * (*ldc)] += a[i + l * (*lda)] * temp;
                }
            }
        }
    } else if (trans_a && !trans_b) {
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
        for (int j = 0; j < *n; j++) {
            for (int i = 0; i < *m; i++) {
                double temp = 0.0;
                for (int l = 0; l < *k; l++) {
                    temp += a[i + l * (*lda)] * b[j + l * (*ldb)];
                }
                c[i + j * (*ldc)] += *alpha * temp;
            }
        }
    } else {
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
    
    if (*m <= 0 || *n <= 0) {
        return;
    }
    
    if (*alpha == 0.0) {
        for (int j = 0; j < *n; j++) {
            for (int i = 0; i < *m; i++) {
                b[i + j * (*ldb)] = 0.0;
            }
        }
        return;
    }
    
    bool unit = (*diag == 'U' || *diag == 'u');
    
    if (*side == 'L' || *side == 'l') {
        if (*transa == 'N' || *transa == 'n') {
            if (*uplo == 'L' || *uplo == 'l') {
                for (int j = 0; j < *n; j++) {
                    if (*alpha != 1.0) {
                        for (int i = 0; i < *m; i++) {
                            b[i + j * (*ldb)] *= *alpha;
                        }
                    }
                    
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
            else {
                for (int j = 0; j < *n; j++) {
                    if (*alpha != 1.0) {
                        for (int i = 0; i < *m; i++) {
                            b[i + j * (*ldb)] *= *alpha;
                        }
                    }
                    
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
        else {
            if (*uplo == 'L' || *uplo == 'l') {
                for (int j = 0; j < *n; j++) {
                    if (*alpha != 1.0) {
                        for (int i = 0; i < *m; i++) {
                            b[i + j * (*ldb)] *= *alpha;
                        }
                    }
                    
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
            else {
                for (int j = 0; j < *n; j++) {
                    if (*alpha != 1.0) {
                        for (int i = 0; i < *m; i++) {
                            b[i + j * (*ldb)] *= *alpha;
                        }
                    }
                    
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
    else {
        // TODO: Implementation of right-side triangular solve
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
        for (int i = 0; i < n; ++i) {
            tmp = *(a + i);
            *(a + i) = *(b + i);
            *(b + i) = tmp;
        }
    } else {
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
        
        #if 0
        if (std::abs(*tmp) <= critere) {
            (*tmp) = critere;
            (*nbpivot)++;
        }
        #endif
        
        tmp1 = tmp + 1;
        int tmp_dim = n - k - 1;
        double sca_tmp = one / (*tmp);
        
        dscal(&tmp_dim, &sca_tmp, tmp1, &iun);
        
        // symmetric rank-1 update of the trailing submatrix
        int dimx = n - k - 1;
        double diag = -(*tmp);
        double *tmp1_stride = tmp1 + stride;
        
        char uplo = 'L';
        dsyr(&uplo, &dimx, &diag, tmp1, &iun, tmp1_stride, &stride);
    }
}

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
    
    const int LAPACK_ROW_MAJOR = 101;
    const int LAPACK_COL_MAJOR = 102;
    
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
    
    if (LAPACK_ROW_MAJOR == matrix_layout) {
        int ldx_t = embedded_maximum(1, m);
        if (ldx < n) return -6;
        
        std::vector<double> x_t(ldx_t * embedded_maximum(1, n));
        
        transpose_into(x_t.data(), ldx_t, x, ldx, m, n, matrix_layout);
        
        int info = dlapmt(LAPACK_COL_MAJOR, forwrd, m, n, x_t.data(), ldx_t, k);
        if (info < 0) return info;
        
        transpose_into(x, ldx, x_t.data(), ldx_t, m, n, LAPACK_COL_MAJOR);
        
        return 0;
    } 
    else if (LAPACK_COL_MAJOR == matrix_layout) {
        if (ldx < m) return -6;
        
        std::vector<double> temp_col(m);
        std::vector<int> perm(n);
        
        for (int i = 0; i < n; i++) {
            perm[i] = i;
        }
        
        if (forwrd) {
            // Forward permutation
            for (int j = 0; j < n; j++) {
                if (perm[j] == j) continue;
                
                int curr_col = j;
                int dest_col = k[j] - 1;
                
                for (int i = 0; i < m; i++) {
                    temp_col[i] = x[i + curr_col * ldx];
                }
                
                while (dest_col != j) {
                    for (int i = 0; i < m; i++) {
                        x[i + curr_col * ldx] = x[i + dest_col * ldx];
                    }
                    
                    perm[curr_col] = perm[dest_col];
                    
                    curr_col = dest_col;
                    dest_col = k[curr_col] - 1;
                }
                
                for (int i = 0; i < m; i++) {
                    x[i + curr_col * ldx] = temp_col[i];
                }
                
                perm[curr_col] = j;
            }
        } else {
            // Backward permutation
            for (int j = 0; j < n; j++) {
                if (perm[j] == j) continue;
                
                int curr_col = j;
                int dest_col = k[j] - 1;
                
                for (int i = 0; i < m; i++) {
                    temp_col[i] = x[i + curr_col * ldx];
                }
                
                while (dest_col != j) {
                    for (int i = 0; i < m; i++) {
                        x[i + curr_col * ldx] = x[i + dest_col * ldx];
                    }
                    
                    perm[curr_col] = perm[dest_col];
                    
                    curr_col = dest_col;
                    dest_col = k[curr_col] - 1;
                }
                
                for (int i = 0; i < m; i++) {
                    x[i + curr_col * ldx] = temp_col[i];
                }
                
                perm[curr_col] = j;
            }
        }
        
        return 0;
    } else {
        return -1;
    }
}

int dgetrf(int matrix_layout, int m, int n, double *a, int lda, int *ipiv) {
    std::cout << "\n\n**************************************************************" << std::endl;
    std::cout << "*** [EMBEDDED] Using embedded dgetrf implementation (m=" << m << ", n=" << n << ") ***" << std::endl;
    std::cout << "**************************************************************\n\n" << std::flush;
    
    const int LAPACK_ROW_MAJOR = 101;
    const int LAPACK_COL_MAJOR = 102;
    
    if (matrix_layout != LAPACK_ROW_MAJOR && matrix_layout != LAPACK_COL_MAJOR) {
        return -1;
    }
    
    if (m < 0) {
        return -2;
    }
    
    if (n < 0) {
        return -3;
    }
    
    if (matrix_layout == LAPACK_COL_MAJOR && lda < std::max(1, m)) {
        return -5;
    }
    
    if (matrix_layout == LAPACK_ROW_MAJOR && lda < std::max(1, n)) {
        return -5;
    }
    
    if (m == 0 || n == 0) {
        return 0;
    }
    
    int min_mn = std::min(m, n);
    int info = 0;
    
    if (matrix_layout == LAPACK_COL_MAJOR) {
        // LU factorization for column-major format
        for (int k = 0; k < min_mn; k++) {
            // Find pivot - the row with largest absolute value in column k
            int p = k;
            double max_val = std::abs(a[k + k*lda]);
            
            for (int i = k+1; i < m; i++) {
                double val = std::abs(a[i + k*lda]);
                if (val > max_val) {
                    max_val = val;
                    p = i;
                }
            }
            
            // Record pivot index of 1-based
            ipiv[k] = p + 1;
            
            if (a[p + k*lda] == 0.0) {
                if (info == 0) info = k + 1;
                continue;
            }
            
            if (p != k) {
                for (int j = 0; j < n; j++) {
                    std::swap(a[k + j*lda], a[p + j*lda]);
                }
            }
            
            for (int i = k+1; i < m; i++) {
                a[i + k*lda] /= a[k + k*lda];
            }
            
            for (int j = k+1; j < n; j++) {
                for (int i = k+1; i < m; i++) {
                    a[i + j*lda] -= a[i + k*lda] * a[k + j*lda];
                }
            }
        }
    }
    else { // Row-major format
        // LU factorization for row-major format
        for (int k = 0; k < min_mn; k++) {
            int p = k;
            double max_val = std::abs(a[k*lda + k]);
            
            for (int i = k+1; i < m; i++) {
                double val = std::abs(a[i*lda + k]);
                if (val > max_val) {
                    max_val = val;
                    p = i;
                }
            }
            
            ipiv[k] = p + 1;
            
            if (a[p*lda + k] == 0.0) {
                if (info == 0) info = k + 1;
                continue;
            }
            
            if (p != k) {
                for (int j = 0; j < n; j++) {
                    std::swap(a[k*lda + j], a[p*lda + j]);
                }
            }
            
            for (int i = k+1; i < m; i++) {
                a[i*lda + k] /= a[k*lda + k];
            }
            
            for (int i = k+1; i < m; i++) {
                for (int j = k+1; j < n; j++) {
                    a[i*lda + j] -= a[i*lda + k] * a[k*lda + j];
                }
            }
        }
    }
    
    return info;
}

int dsytrf(int matrix_layout, char uplo, int n, double *a, int lda, int *ipiv) {
    std::cout << "\n\n**************************************************************" << std::endl;
    std::cout << "*** [EMBEDDED] Using embedded dsytrf implementation (n=" << n << ", uplo=" << uplo << ") ***" << std::endl;
    std::cout << "**************************************************************\n\n" << std::flush;
    
    const int LAPACK_ROW_MAJOR = 101;
    const int LAPACK_COL_MAJOR = 102;
    
    if (matrix_layout != LAPACK_ROW_MAJOR && matrix_layout != LAPACK_COL_MAJOR) {
        return -1;
    }
    
    if (uplo != 'U' && uplo != 'u' && uplo != 'L' && uplo != 'l') {
        return -2;
    }
    
    if (n < 0) {
        return -3;
    }
    
    if (lda < std::max(1, n)) {
        return -5;
    }
    
    if (n == 0) {
        return 0;
    }
    
    for (int i = 0; i < n; i++) {
        ipiv[i] = i + 1;
    }
    
    std::vector<double> a_copy;
    double *a_ptr = a;
    
    if (matrix_layout == LAPACK_ROW_MAJOR) {
        a_copy.resize(n * lda);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a_copy[j * lda + i] = a[i * lda + j];
            }
        }
        a_ptr = a_copy.data();
    }
    
    bool lower = (uplo == 'L' || uplo == 'l');
    
    int info = 0;
    int nbpivot = 0;
    double critere = 1e-10;
    
    const double alpha = (1.0 + std::sqrt(17.0)) / 8.0;
    
    int k = 0;
    while (k < n) {
        if (lower) {
            double colmax = 0.0;
            int r = -1;
            for (int i = k + 1; i < n; i++) {
                double abs_val = std::abs(a_ptr[i + k * lda]);
                if (abs_val > colmax) {
                    colmax = abs_val;
                    r = i;
                }
            }
            
            double akk = a_ptr[k + k * lda];
            
            if (colmax == 0.0 || std::abs(akk) >= alpha * colmax) {
                if (std::abs(akk) <= critere) {
                    if (info == 0) info = k + 1;
                    akk = (akk > 0) ? critere : -critere;
                    a_ptr[k + k * lda] = akk;
                    nbpivot++;
                }
                
                if (colmax > 0.0) {
                    double d = 1.0 / akk;
                    for (int i = k + 1; i < n; i++) {
                        a_ptr[i + k * lda] *= d;
                    }
                }
                
                for (int j = k + 1; j < n; j++) {
                    double temp = a_ptr[j + k * lda];
                    for (int i = j; i < n; i++) {
                        a_ptr[i + j * lda] -= temp * a_ptr[i + k * lda];
                    }
                }
                
                ipiv[k] = k + 1;
                k++;
            } else {
                if (r == -1) r = k + 1;
                
                double rowmax = 0.0;
                for (int i = k; i < r; i++) {
                    double abs_val = std::abs(a_ptr[r + i * lda]);
                    if (abs_val > rowmax) {
                        rowmax = abs_val;
                    }
                }
                for (int i = r + 1; i < n; i++) {
                    double abs_val = std::abs(a_ptr[i + r * lda]);
                    if (abs_val > rowmax) {
                        rowmax = abs_val;
                    }
                }
                
                double arr = a_ptr[r + r * lda];
                
                if (rowmax == 0.0 || std::abs(arr) >= alpha * rowmax) {
                    ipiv[k] = r + 1;
                    
                    if (r != k) {
                        std::swap(a_ptr[k + k * lda], a_ptr[r + r * lda]);
                        
                        for (int i = 0; i < k; i++) {
                            std::swap(a_ptr[i + k * lda], a_ptr[i + r * lda]);
                        }
                        
                        for (int j = k + 1; j < r; j++) {
                            std::swap(a_ptr[k + j * lda], a_ptr[j + r * lda]);
                        }
                        
                        for (int i = r + 1; i < n; i++) {
                            std::swap(a_ptr[i + k * lda], a_ptr[i + r * lda]);
                        }
                    }
                    
                    double akk = a_ptr[k + k * lda];
                    
                    if (std::abs(akk) <= critere) {
                        if (info == 0) info = k + 1;
                        akk = (akk > 0) ? critere : -critere;
                        a_ptr[k + k * lda] = akk;
                        nbpivot++;
                    }
                    
                    if (colmax > 0.0) {
                        double d = 1.0 / akk;
                        for (int i = k + 1; i < n; i++) {
                            a_ptr[i + k * lda] *= d;
                        }
                    }
                    
                    for (int j = k + 1; j < n; j++) {
                        double temp = a_ptr[j + k * lda];
                        for (int i = j; i < n; i++) {
                            a_ptr[i + j * lda] -= temp * a_ptr[i + k * lda];
                        }
                    }
                    
                    k++;
                } else {
                    ipiv[k] = k + 1;
                    k++;
                }
            }
        } else {
            double rowmax = 0.0;
            int r = -1;
            for (int j = k + 1; j < n; j++) {
                double abs_val = std::abs(a_ptr[k + j * lda]);
                if (abs_val > rowmax) {
                    rowmax = abs_val;
                    r = j;
                }
            }
            
            double akk = a_ptr[k + k * lda];
            
            if (rowmax == 0.0 || std::abs(akk) >= alpha * rowmax) {
                if (std::abs(akk) <= critere) {
                    if (info == 0) info = k + 1;
                    akk = (akk > 0) ? critere : -critere;
                    a_ptr[k + k * lda] = akk;
                    nbpivot++;
                }
                
                if (rowmax > 0.0) {
                    double d = 1.0 / akk;
                    for (int j = k + 1; j < n; j++) {
                        a_ptr[k + j * lda] *= d;
                    }
                }
                
                for (int i = k + 1; i < n; i++) {
                    double temp = a_ptr[k + i * lda];
                    for (int j = i; j < n; j++) {
                        a_ptr[i + j * lda] -= temp * a_ptr[k + j * lda];
                    }
                }
                
                ipiv[k] = k + 1;
                k++;
            } else {
                ipiv[k] = k + 1;
                k++;
            }
        }
    }
    
    if (matrix_layout == LAPACK_ROW_MAJOR) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a[i * lda + j] = a_copy[j * lda + i];
            }
        }
    }
    
    return info;
}

} // namespace embedded
} // namespace nasoq 