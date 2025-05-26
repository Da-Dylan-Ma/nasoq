//
// Created by kazem on 11/29/18.
//

#ifndef PROJECT_BLASKERNELS_H
#define PROJECT_BLASKERNELS_H

#include <cstdint>
#include <vector>
#include <cstdlib>
#include "nasoq/common/Reach.h"

// Debug preprocessor macros
#ifdef EMBEDDED
#pragma message("EMBEDDED macro is defined!")
#else
#pragma message("EMBEDDED macro is NOT defined!")
#endif

// Include embedded implementation headers if we're using them
#ifdef EMBEDDED
#include "nasoq/embedded/embedded_blas.h"

// Define LAPACK constants for embedded mode
#define LAPACK_ROW_MAJOR 101
#define LAPACK_COL_MAJOR 102
#endif

// Include BLAS headers based on the selected backend
#if !defined(EMBEDDED) || defined(EMBEDDED_WITH_BLAS)
  #ifdef OPENBLAS
    /*    #ifdef OB_INTERNAL
      #include "lapacke.h"
      #include "cblas.h"
      //#include "common_interface.h"
      #else*/
      //#include "openblas/f77blas.h"
    #ifdef NASOQ_USE_CLAPACK
      #include "nasoq/clapacke/clapacke.h"
    #else
      #include "openblas/lapacke.h"
    #endif
      #include "openblas/cblas.h"
    // #endif
  #else
    #include "mkl.h"
    #include <mkl_blas.h>
    #include <mkl_lapacke.h>
  #endif
#endif

namespace nasoq {
#  define VEC_SCAL(n, a, x, u){               \
    int i; double *pt,*p=(x);                    \
    for((pt)=(p+n);(p)<(pt);)                   \
      *((p)++)*= (a);                           \
  }

// Define the BLAS function macros
#ifdef EMBEDDED
  // Use embedded implementations
  #define SYM_DSCAL nasoq::embedded::dscal
  #define SYM_DSYR nasoq::embedded::dsyr
  #define SYM_DCOPY nasoq::embedded::dcopy
  #define SYM_DGEMV nasoq::embedded::dgemv
  #define SYM_DGEMM nasoq::embedded::dgemm
  #define SYM_DTRSM nasoq::embedded::dtrsm
  #define NASOQ_DOT nasoq::embedded::dot
  #define NASOQ_SWAP_VECTOR nasoq::embedded::swap_vector
  #define SYM_SYTRF nasoq::embedded::sym_sytrf
  #define NASOQ_DLAPMT nasoq::embedded::dlapmt
  #define NASOQ_DGETRF nasoq::embedded::dgetrf
  #define NASOQ_DSYTRF nasoq::embedded::dsytrf
  #define SET_BLAS_THREAD(t) ((void)0) // No-op for embedded
  
  // Alias LAPACKE functions to our embedded implementations
  #define LAPACKE_dsytrf nasoq::embedded::dsytrf
  #define LAPACKE_dlapmt nasoq::embedded::dlapmt
  #define LAPACKE_dgetrf nasoq::embedded::dgetrf
#else
  // Use external BLAS
  #ifdef OPENBLAS
    #define SYM_DGEMM dgemm_
    #define SYM_DTRSM dtrsm_
    #define SYM_DGEMV dgemv_
    #define SYM_DSYR dsyr_
    #define SYM_DCOPY dcopy_
    #define SYM_DSCAL dscal_
    #define SET_BLAS_THREAD(t) (openblas_set_num_threads(t))
  #else
    #define SYM_DGEMM dgemm
    #define SYM_DTRSM dtrsm
    #define SYM_DGEMV dgemv
    #define SYM_DSYR dsyr
    #define SYM_DCOPY dcopy
    #define SYM_DSCAL dscal
    #define SET_BLAS_THREAD(t) (MKL_Domain_Set_Num_Threads(t, MKL_DOMAIN_BLAS))
  #endif
  
  // External implementation definitions
  #define NASOQ_DOT nasoq::dot
  #define NASOQ_SWAP_VECTOR nasoq::swap_vector
  #define SYM_SYTRF nasoq::sym_sytrf
  #define NASOQ_DLAPMT nasoq::clapacke::LAPACKE_dlapmt
  #define NASOQ_DGETRF nasoq::clapacke::LAPACKE_dgetrf
  #define NASOQ_DSYTRF LAPACKE_dsytrf
#endif

 void
 sym_sytrf(double *A, int n, const int stride, int *nbpivot,
           double critere);

 double dot(int n, double *a, double *b);


/*
 * swaps a and b
 */
 void swap_vector(int n, double *a, double *b, int lda);

 void swap_int(int &a, int &b);

/*
 * Performs reverse ordering after sytrf
 * assume cur has full lower triangular.
 */
 int reorder_after_sytrf(int n, double *a, int lda, int *ipiv,
                         int *perm, double *D, int lda_D,
                         int *swap_vec, int *ws);

/*
 * Shoft rows up/down in a block of CSC, used in the next func
 */
 void shift(int n_col, int n_row, double *a, int lda, const int inc);

/*
 * This function re-order the part of L facing to diagonal
 * according to given ordering and swap array.
 */
 void swapping_row_in_blockedL_new(int supWdt, int supNo, const size_t *lC, const size_t *Li_ptr, int *lR,
                                   const int curCol, const int *blockSet, double *lValues,
                                   int top, int *xi, int *lbs, int *ubs,
                                   int *cur_perm, int *cur_swap);


/*
 * Row reordering in blocked L matrix
 */
 void row_reordering(int supNo, const size_t *lC, const size_t *Li_ptr, int *lR,
                     const int *blockSet, int *aTree, int *cT, int *rT, int *col2Sup, double *lValues,
                     std::vector<int> swap_req, int *full_swap, int *xi, int *map, int *ws, double *wsf);

/*
 * The input is blocked diagonal matrix, the max block size is 2x2
 * Suitable for LDLT SBK .
 * Diagonal matrix is stored in array of size 2*n
 */

 void blocked_2by2_solver(int n, double *D, double *rhs,
                          int n_rhs, int lda, int lda_d);

/*
 * The input is blocked diagonal matrix, the max block size is 2x2
 * Suitable for LDLT SBK .
 * Diagonal matrix is stored in array of size 2*n
 */

 void blocked_2by2_solver_update(int n, double *D, double *rhs,
                                 int n_rhs, int lda, int lda_d, int *mask);

/*
 * The input is blocked diagonal matrix, the max block size is 2x2
 * Suitable for LDLT SBK .
 * Diagonal matrix is stored in array of size 2*n
 * diagonal matrix is symmetric
 */

 void blocked_2by2_mult(int n, int m, double *D, double *src, double *dst,
                        int lda, int lda_d);
}
#endif //PROJECT_BLASKERNELS_H
