/**
 * @file embedded_blas.h
 * @brief Embedded-friendly implementations of common BLAS routines
 * 
 * This file contains implementations of BLAS routines that do not rely on
 * external libraries. These are useful for embedded platforms where 
 * standard BLAS libraries may not be available.
 */

#ifndef NASOQ_EMBEDDED_BLAS_H
#define NASOQ_EMBEDDED_BLAS_H

namespace nasoq {
namespace embedded {

/**
 * @brief Vector scaling (DSCAL)
 * 
 * Computes x = alpha*x where x is a vector
 * 
 * @param n     Number of elements in vector x
 * @param alpha Scaling factor
 * @param x     Vector to be scaled (modified in-place)
 * @param incx  Stride between consecutive elements of x
 */
void dscal(const int *n, const double *alpha, double *x, const int *incx);

/**
 * @brief Symmetric rank-1 update (DSYR)
 * 
 * Performs the symmetric rank-1 update: A = alpha*x*x' + A
 * 
 * @param uplo  'L' or 'U' (lower or upper triangular part of A is updated)
 * @param n     Order of matrix A
 * @param alpha Scalar multiplier
 * @param x     Vector x
 * @param incx  Stride between consecutive elements of x
 * @param a     Matrix A (modified in-place)
 * @param lda   Leading dimension of A
 */
void dsyr(const char *uplo, const int *n, const double *alpha,
          const double *x, const int *incx, double *a, const int *lda);

/**
 * @brief Vector copy (DCOPY)
 * 
 * Copies vector x to vector y
 * 
 * @param n     Number of elements to copy
 * @param x     Source vector
 * @param incx  Stride between consecutive elements of x
 * @param y     Destination vector
 * @param incy  Stride between consecutive elements of y
 */
void dcopy(const int *n, const double *x, const int *incx, 
           double *y, const int *incy);

/**
 * @brief Vector dot product (DDOT)
 * 
 * Computes the dot product of two vectors: result = sum(x[i] * y[i])
 * 
 * @param n     Number of elements in vectors
 * @param a     First vector
 * @param b     Second vector
 * @return      Dot product of a and b
 */
double dot(int n, const double *a, const double *b);

/**
 * @brief Vector swap (DSWAP)
 * 
 * Swaps the contents of two vectors
 * 
 * @param n     Number of elements to swap
 * @param a     First vector
 * @param b     Second vector
 * @param lda   Stride between consecutive elements (can be 1 for regular stride or other values)
 */
void swap_vector(int n, double *a, double *b, int lda);

/**
 * @brief Solver for block-diagonal matrix with 1x1 and 2x2 blocks
 * 
 * Solves D*X = B where D is a block diagonal matrix with 1x1 and 2x2 blocks
 * 
 * @param n     Order of matrix D
 * @param D     Block diagonal matrix
 * @param rhs   Right-hand-side vectors (solution returned here)
 * @param n_rhs Number of right-hand sides
 * @param lda   Leading dimension of rhs
 * @param lda_d Stride for subdiagonal elements in D
 */
void blocked_2by2_solver(int n, double *D, double *rhs, int n_rhs, int lda, int lda_d);

/**
 * @brief Matrix-vector multiplication (DGEMV)
 * 
 * Performs one of the matrix-vector operations:
 * y = alpha*A*x + beta*y  (trans = 'N' or 'n')
 * y = alpha*A'*x + beta*y (trans = 'T', 't', 'C', or 'c')
 * 
 * @param trans  Specifies operation to perform ('N', 'n', 'T', 't', 'C', 'c')
 * @param m      Number of rows of matrix A
 * @param n      Number of columns of matrix A
 * @param alpha  Scalar multiplier for A*x or A'*x
 * @param a      Matrix A
 * @param lda    Leading dimension of A
 * @param x      Vector x
 * @param incx   Stride between consecutive elements of x
 * @param beta   Scalar multiplier for y
 * @param y      Vector y (modified in-place)
 * @param incy   Stride between consecutive elements of y
 */
void dgemv(const char *trans, const int *m, const int *n,
           const double *alpha, const double *a, const int *lda,
           const double *x, const int *incx,
           const double *beta, double *y, const int *incy);

/**
 * @brief Multiplication with block-diagonal matrix with 1x1 and 2x2 blocks
 * 
 * Computes dst = D*src where D is a block diagonal matrix with 1x1 and 2x2 blocks
 * 
 * @param n     Order of matrix D
 * @param m     Number of columns in src/dst
 * @param D     Block diagonal matrix
 * @param src   Source matrix
 * @param dst   Destination matrix (result)
 * @param lda   Leading dimension of src
 * @param lda_d Stride for subdiagonal elements in D
 */
void blocked_2by2_mult(int n, int m, double *D, double *src, double *dst, int lda, int lda_d);

/**
 * @brief Matrix-matrix multiplication (DGEMM)
 * 
 * Performs one of the matrix-matrix operations:
 * C = alpha*A*B + beta*C  (transA='N', transB='N')
 * C = alpha*A'*B + beta*C (transA='T', transB='N')
 * C = alpha*A*B' + beta*C (transA='N', transB='T')
 * C = alpha*A'*B' + beta*C (transA='T', transB='T')
 * 
 * @param transa Specifies if A should be transposed ('N', 'n', 'T', 't', 'C', 'c')
 * @param transb Specifies if B should be transposed ('N', 'n', 'T', 't', 'C', 'c')
 * @param m      Number of rows of matrix C and op(A)
 * @param n      Number of columns of matrix C and op(B)
 * @param k      Number of columns of op(A) and rows of op(B)
 * @param alpha  Scalar multiplier for op(A)*op(B)
 * @param a      Matrix A
 * @param lda    Leading dimension of A
 * @param b      Matrix B
 * @param ldb    Leading dimension of B
 * @param beta   Scalar multiplier for C
 * @param c      Matrix C (modified in-place)
 * @param ldc    Leading dimension of C
 */
void dgemm(const char *transa, const char *transb, const int *m, const int *n, const int *k,
           const double *alpha, const double *a, const int *lda, const double *b, const int *ldb,
           const double *beta, double *c, const int *ldc);

/**
 * @brief Triangular matrix solve with multiple right-hand sides (DTRSM)
 * 
 * Solves one of the matrix equations:
 * op(A)*X = alpha*B (side='L' or 'l')
 * X*op(A) = alpha*B (side='R' or 'r')
 * 
 * where op(A) = A or A', A is a triangular matrix, and X and B are m-by-n matrices.
 * 
 * @param side   Specifies whether op(A) is on the left or right of X ('L', 'l', 'R', 'r')
 * @param uplo   Specifies whether A is upper or lower triangular ('U', 'u', 'L', 'l')
 * @param transa Specifies whether to use A or A' ('N', 'n', 'T', 't', 'C', 'c')
 * @param diag   Specifies whether A is unit triangular ('U', 'u', 'N', 'n')
 * @param m      Number of rows of matrix B
 * @param n      Number of columns of matrix B
 * @param alpha  Scalar multiplier for B
 * @param a      Triangular matrix A
 * @param lda    Leading dimension of A
 * @param b      Matrix B on entry, matrix X on exit
 * @param ldb    Leading dimension of B
 */
void dtrsm(const char *side, const char *uplo, const char *transa, const char *diag,
           const int *m, const int *n, const double *alpha,
           const double *a, const int *lda, double *b, const int *ldb);

/**
 * @brief Symmetric factorization (SYTRF)
 * 
 * Computes the factorization of a symmetric matrix A using the
 * LDL^T factorization where L is a lower triangular matrix with 
 * unit diagonal and D is a diagonal matrix.
 * 
 * @param A       The input matrix A (modified in-place to contain the factorization)
 * @param n       The order of the matrix A
 * @param stride  The leading dimension of A
 * @param nbpivot Pivot counter (incremented if small pivots are encountered)
 * @param critere Threshold for detecting small pivots
 */
void sym_sytrf(double *A, int n, const int stride, int *nbpivot, double critere);

/**
 * @brief Permute columns of a matrix (DLAPMT)
 * 
 * Rearranges the columns of the M by N matrix X as specified
 * by the permutation K(1),K(2),...,K(N) of the integers 1,...,N.
 * 
 * @param matrix_layout  Layout of matrix (LAPACK_ROW_MAJOR or LAPACK_COL_MAJOR)
 * @param forwrd         If true, forward permutation: X(*,K(J)) is moved to X(*,J)
 *                       If false, backward permutation: X(*,J) is moved to X(*,K(J))
 * @param m              Number of rows of matrix X
 * @param n              Number of columns of matrix X
 * @param x              Matrix X (modified in-place)
 * @param ldx            Leading dimension of matrix X
 * @param k              Permutation vector
 * @return               0 if successful, negative error code otherwise
 */
int dlapmt(int matrix_layout, int forwrd, int m, int n, 
           double *x, int ldx, int *k);

/**
 * @brief LU factorization (DGETRF)
 * 
 * Computes an LU factorization of a general M-by-N matrix A using partial
 * pivoting with row interchanges.
 * 
 * The factorization has the form A = P * L * U where P is a permutation matrix,
 * L is lower triangular with unit diagonal elements, and U is upper triangular.
 * 
 * @param matrix_layout  Layout of matrix (LAPACK_ROW_MAJOR or LAPACK_COL_MAJOR)
 * @param m              Number of rows of the matrix A
 * @param n              Number of columns of the matrix A
 * @param a              Matrix A (modified in-place to contain L and U)
 * @param lda            Leading dimension of A
 * @param ipiv           Pivot indices (modified in-place)
 * @return               0 if successful, negative error code if an argument had an illegal value,
 *                       positive value i if U(i,i) is exactly zero (matrix is singular)
 */
int dgetrf(int matrix_layout, int m, int n, double *a, int lda, int *ipiv);

} // namespace embedded
} // namespace nasoq

#endif // NASOQ_EMBEDDED_BLAS_H