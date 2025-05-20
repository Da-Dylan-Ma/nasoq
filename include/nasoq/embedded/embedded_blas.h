/**
 * @file embedded_blas.h
 * @brief Embedded-friendly implementations of BLAS functions.
 *
 * This file provides implementations of core BLAS functions that can run
 * on embedded systems without external library dependencies. The implementations
 * prioritize correctness and minimal memory usage over performance.
 */

#ifndef NASOQ_EMBEDDED_BLAS_H
#define NASOQ_EMBEDDED_BLAS_H

namespace nasoq {
namespace embedded {

/**
 * @brief Scale a vector by a constant.
 *
 * Computes x = alpha * x
 *
 * @param n     Pointer to the number of elements in vector x
 * @param alpha Pointer to the scaling factor
 * @param x     Vector to be scaled (modified in-place)
 * @param incx  Pointer to the stride between consecutive elements of x
 *
 * @note Equivalent to BLAS dscal function
 */
void dscal(const int *n, const double *alpha, double *x, const int *incx);

/**
 * @brief Performs a symmetric rank-1 update.
 *
 * Computes A := alpha*x*x' + A, where A is a symmetric matrix.
 * Only the lower or upper triangular part of A is referenced and updated.
 *
 * @param uplo       Pointer to a character indicating whether to update the upper ('U') 
 *                   or lower ('L') triangular part of A
 * @param n          Pointer to the order of matrix A
 * @param alpha      Pointer to the scalar multiplier
 * @param x          Vector of length at least (1 + (n-1)*abs(incx))
 * @param incx       Pointer to the stride between consecutive elements of x
 * @param a          Matrix A, stored as a linear array in column-major order
 * @param lda        Pointer to the leading dimension of A as declared in the calling program
 *
 * @note Equivalent to BLAS dsyr function
 */
void dsyr(const char *uplo, const int *n, const double *alpha,
          const double *x, const int *incx, double *a, const int *lda);

/**
 * @brief Copies a vector to another vector.
 *
 * Computes y = x
 *
 * @param n     Pointer to the number of elements in vectors x and y
 * @param x     Source vector
 * @param incx  Pointer to the stride between consecutive elements of x
 * @param y     Destination vector
 * @param incy  Pointer to the stride between consecutive elements of y
 *
 * @note Equivalent to BLAS dcopy function
 */
void dcopy(const int *n, const double *x, const int *incx,
           double *y, const int *incy);

/**
 * @brief Solves a system with a 2x2 block diagonal matrix.
 *
 * This function handles both 1x1 and 2x2 block pivots in the diagonal D.
 * For 1x1 blocks, it simply scales the corresponding row in rhs.
 * For 2x2 blocks, it solves a small linear system using Cramer's rule.
 *
 * @param n     Number of columns in D
 * @param D     The block diagonal matrix
 * @param rhs   Right-hand side matrix (overwritten with solution)
 * @param n_rhs Number of right-hand sides
 * @param lda   Leading dimension of rhs
 * @param lda_d Leading dimension of D
 *
 * @note Specialized function used by NASOQ
 */
void blocked_2by2_solver(int n, double *D, double *rhs, int n_rhs, int lda, int lda_d);

/**
 * @brief Multiplies a 2x2 block diagonal matrix by a matrix.
 *
 * This function multiplies a block diagonal matrix with a source matrix.
 * It handles both 1x1 and 2x2 blocks in the diagonal matrix D.
 *
 * @param n     Number of columns in D
 * @param m     Number of columns in src and dst
 * @param D     The block diagonal matrix
 * @param src   Source matrix
 * @param dst   Destination matrix (output of the multiplication)
 * @param lda   Leading dimension of src and dst
 * @param lda_d Leading dimension of D
 *
 * @note Specialized function used by NASOQ
 */
void blocked_2by2_mult(int n, int m, double *D, double *src, double *dst, int lda, int lda_d);

} // namespace embedded
} // namespace nasoq

#endif // NASOQ_EMBEDDED_BLAS_H