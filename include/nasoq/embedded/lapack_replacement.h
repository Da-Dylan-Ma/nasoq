/**
 * @file lapack_replacement.h
 * @brief Replacement functions for LAPACK routines for bare-metal ARM Cortex implementation
 */

#ifndef NASOQ_EMBEDDED_LAPACK_REPLACEMENT_H
#define NASOQ_EMBEDDED_LAPACK_REPLACEMENT_H

namespace nasoq {
namespace embedded {

/**
 * @brief Computes the factorization of a symmetric matrix using diagonal pivoting.
 * Replacement for DSYTRF.
 * 
 * @param uplo Specifies whether the upper or lower triangular part is stored:
 *             'U' or 'u' for upper triangular, 'L' or 'l' for lower triangular
 * @param n The order of the matrix A
 * @param a Array containing the matrix A. On exit, contains the block diagonal
 *        matrix D and the multipliers used to obtain the factor U or L.
 * @param lda The leading dimension of the array a
 * @param ipiv Array of dimension (n). Details of the interchanges and the block
 *        structure of D.
 * @param work Array of dimension (lwork). On exit, if info = 0, work(1) returns the optimal
 *        lwork.
 * @param lwork The length of work. lwork >= max(1,n)
 * @param info If 0, successful exit. If <0, the i-th argument had an illegal value.
 *        If >0, D(i,i) is exactly zero. The factorization has been completed, but the
 *        block diagonal matrix D is exactly singular.
 */
void dsytrf(const char* uplo, const int* n, double* a, const int* lda,
            int* ipiv, double* work, const int* lwork, int* info);

/**
 * @brief Performs a matrix permutation. Replacement for LAPMT.
 * 
 * @param forwrd Specifies the direction of permutation:
 *               = true  forward permutation
 *               = false backward permutation
 * @param m Number of rows of the matrix A
 * @param n Number of columns of the matrix A
 * @param a Array containing the matrix A
 * @param lda The leading dimension of the array a
 * @param perm Array of dimension (n) that defines the permutation
 */
void lapmt(const bool* forwrd, const int* m, const int* n,
           double* a, const int* lda, const int* perm);

} // namespace embedded
} // namespace nasoq

#endif // NASOQ_EMBEDDED_LAPACK_REPLACEMENT_H 