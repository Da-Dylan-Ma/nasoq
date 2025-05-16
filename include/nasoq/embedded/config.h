/**
 * @file config.h
 * @brief Configuration for NASOQ embedded implementation
 */

#ifndef NASOQ_EMBEDDED_CONFIG_H
#define NASOQ_EMBEDDED_CONFIG_H

// Define NASOQ_EMBEDDED to enable the embedded implementation
// #define NASOQ_EMBEDDED

// Define LAPACK constants that would normally be included from LAPACK headers
#define LAPACK_ROW_MAJOR 101
#define LAPACK_COL_MAJOR 102

// Equivalent BLAS function names
#ifdef NASOQ_EMBEDDED
  #include "nasoq/embedded/blas_replacement.h"
  #include "nasoq/embedded/lapack_replacement.h"
  #include "nasoq/embedded/custom_blas.h"
  
  // BLAS functions
  #undef SYM_DGEMM
  #undef SYM_DTRSM
  #undef SYM_DGEMV
  #undef SYM_DSCAL
  
  #define SYM_DGEMM nasoq::embedded::dgemm
  #define SYM_DTRSM nasoq::embedded::dtrsm
  #define SYM_DGEMV nasoq::embedded::dgemv
  #define SYM_DSCAL nasoq::embedded::dscal
  #define SYM_DSYR  nasoq::embedded::dsyr
  
  // LAPACK functions - simple replacements using inline functions
  inline int nasoq_embedded_dsytrf(int layout, char uplo, int n, double* a, int lda, int* ipiv) {
    int info = 0;
    // Use a static work array for efficiency and to avoid memory leaks
    static double* work = nullptr;
    static int work_size = 0;
    
    // Only allocate if needed or if the size needs to increase
    if (work == nullptr || work_size < n) {
      delete[] work; // Safe even if work is nullptr
      work_size = n;
      work = new double[work_size];
    }
    
    const char uplo_str[] = {uplo, '\0'};
    nasoq::embedded::dsytrf(uplo_str, &n, a, &lda, ipiv, work, &n, &info);
    
    // Don't delete work here - it will be reused
    return info;
  }
  
  inline int nasoq_embedded_dlapmt(int layout, int forwrd, int m, int n, double* a, int lda, int* perm) {
    bool forwrd_bool = (forwrd != 0);
    nasoq::embedded::lapmt(&forwrd_bool, &m, &n, a, &lda, perm);
    return 0;
  }
  
  #define LAPACKE_dsytrf nasoq_embedded_dsytrf
  #define LAPACKE_dlapmt nasoq_embedded_dlapmt
  
  // Custom BLAS functions
  #define dlsolve_blas_nonUnit nasoq::embedded::dlsolve_blas_nonUnit
  #define lSolve_dense_col_sync nasoq::embedded::lSolve_dense_col_sync
  #define dmatvec_blas nasoq::embedded::dmatvec_blas
  
  // Empty definition for thread control (no-op in the embedded version)
  #define SET_BLAS_THREAD(t)
#endif

#endif // NASOQ_EMBEDDED_CONFIG_H 