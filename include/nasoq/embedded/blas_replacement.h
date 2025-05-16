/**
 * @file blas_replacement.h
 * @brief Replacement functions for BLAS routines for bare-metal ARM Cortex implementation
 */

#ifndef NASOQ_EMBEDDED_BLAS_REPLACEMENT_H
#define NASOQ_EMBEDDED_BLAS_REPLACEMENT_H

namespace nasoq {
namespace embedded {

/**
 * @brief Matrix-matrix multiplication replacement for DGEMM
 * C := alpha*op(A)*op(B) + beta*C
 * 
 * @param transa Specifies op(A): 'N'/'n' for A, 'T'/'t' for A^T
 * @param transb Specifies op(B): 'N'/'n' for B, 'T'/'t' for B^T
 * @param m Number of rows of matrix op(A) and C
 * @param n Number of columns of matrix op(B) and C
 * @param k Number of columns of op(A) and rows of op(B)
 * @param alpha Scalar multiplier for op(A)*op(B)
 * @param a Matrix A
 * @param lda Leading dimension of A
 * @param b Matrix B
 * @param ldb Leading dimension of B
 * @param beta Scalar multiplier for C
 * @param c Matrix C (result)
 * @param ldc Leading dimension of C
 */
void dgemm(const char* transa, const char* transb,
           const int* m, const int* n, const int* k,
           const double* alpha, const double* a, const int* lda,
           const double* b, const int* ldb,
           const double* beta, double* c, const int* ldc);

/**
 * @brief Triangular solve with multiple right-hand sides replacement for DTRSM
 * op(A)*X = alpha*B or X*op(A) = alpha*B
 * 
 * @param side Specifies whether op(A) is on the left ('L'/'l') or right ('R'/'r') of X
 * @param uplo Specifies whether A is upper ('U'/'u') or lower ('L'/'l') triangular
 * @param transa Specifies op(A): 'N'/'n' for A, 'T'/'t' for A^T
 * @param diag Specifies if A is unit triangular ('U'/'u') or not ('N'/'n')
 * @param m Number of rows of B
 * @param n Number of columns of B
 * @param alpha Scalar multiplier for B
 * @param a Matrix A
 * @param lda Leading dimension of A
 * @param b Matrix B (on entry) and X (on exit)
 * @param ldb Leading dimension of B
 */
void dtrsm(const char* side, const char* uplo,
           const char* transa, const char* diag,
           const int* m, const int* n,
           const double* alpha, const double* a, const int* lda,
           double* b, const int* ldb);

/**
 * @brief Matrix-vector multiplication replacement for DGEMV
 * y := alpha*op(A)*x + beta*y
 * 
 * @param trans Specifies op(A): 'N'/'n' for A, 'T'/'t' for A^T
 * @param m Number of rows of A
 * @param n Number of columns of A
 * @param alpha Scalar multiplier for op(A)*x
 * @param a Matrix A
 * @param lda Leading dimension of A
 * @param x Vector x
 * @param incx Increment for the elements of x
 * @param beta Scalar multiplier for y
 * @param y Vector y (result)
 * @param incy Increment for the elements of y
 */
void dgemv(const char* trans,
           const int* m, const int* n,
           const double* alpha, const double* a, const int* lda,
           const double* x, const int* incx,
           const double* beta, double* y, const int* incy);

/**
 * @brief Vector scaling replacement for DSCAL
 * x := alpha*x
 * 
 * @param n Number of elements in vector x
 * @param alpha Scalar multiplier
 * @param x Vector to be scaled
 * @param incx Increment for the elements of x
 */
void dscal(const int* n, const double* alpha,
           double* x, const int* incx);

/**
 * @brief Symmetric rank-1 update replacement for DSYR
 * A := alpha*x*x^T + A
 * 
 * @param uplo Specifies whether the upper ('U'/'u') or lower ('L'/'l') part of A is used
 * @param n Order of matrix A
 * @param alpha Scalar multiplier
 * @param x Vector x
 * @param incx Increment for the elements of x
 * @param a Matrix A
 * @param lda Leading dimension of A
 */
void dsyr(const char* uplo, const int* n,
          const double* alpha, const double* x, const int* incx,
          double* a, const int* lda);

} // namespace embedded
} // namespace nasoq

#endif // NASOQ_EMBEDDED_BLAS_REPLACEMENT_H 