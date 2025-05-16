/**
 * @file custom_blas.h
 * @brief Custom BLAS-like functions for NASOQ embedded implementation
 */

#ifndef NASOQ_EMBEDDED_CUSTOM_BLAS_H
#define NASOQ_EMBEDDED_CUSTOM_BLAS_H

namespace nasoq {
namespace embedded {

/**
 * @brief Non-unit lower triangular solve with a single right-hand side.
 * Solves L * x = b for x, where L is a lower triangular matrix stored in column-major format.
 * 
 * @param ldm Leading dimension of L
 * @param ncol Number of columns in L (should be equal to number of rows)
 * @param M Matrix L in column-major format
 * @param rhs Right-hand side b on entry, solution x on exit
 */
void dlsolve_blas_nonUnit(int ldm, int ncol, double *M, double *rhs);

/**
 * @brief Row-synchronized lower triangular solve for a column of the right-hand side.
 * 
 * @param colSize The number of rows in the triangular matrix
 * @param col The column index to solve for
 * @param M The triangular matrix in column-major format
 * @param rhs The right-hand side on entry, solution on exit
 */
void lSolve_dense_col_sync(int colSize, int col, double *M, double *rhs);

/**
 * @brief Matrix-vector multiplication: Mxvec += M * vec
 * 
 * @param ldm Leading dimension of M
 * @param nrow Number of rows in the submatrix of M to use
 * @param ncol Number of columns in the submatrix of M to use
 * @param M Matrix M in column-major format
 * @param vec Input vector
 * @param Mxvec Result vector (accumulates the product)
 */
void dmatvec_blas(int ldm, int nrow, int ncol, double *M, double *vec, double *Mxvec);

} // namespace embedded
} // namespace nasoq

#endif // NASOQ_EMBEDDED_CUSTOM_BLAS_H 