//
// Created by kazem on 11/5/18.
//

#ifndef PROJECT_SERIAL_BLOCKED_LDL_H
#define PROJECT_SERIAL_BLOCKED_LDL_H

#include <cstddef>
#include <vector>

namespace nasoq{

    static void eigen_dgemm(const char* /*transA*/, const char* /*transB*/,
                            const int *M, const int *N, const int *K,
                            const double *alpha,
                            const double *A, const int *lda,
                            const double *B, const int *ldb,
                            const double *beta,
                            double *C, const int *ldc);

    static void eigen_dtrsm(const char* /*side*/, const char* /*uplo*/,
                            const char* /*transA*/, const char* /*diag*/,
                            const int *M, const int *N,
                            const double *alpha,
                            const double *A, const int *lda,
                            double *B, const int *ldb);

    static void eigen_sym_sytrf(double* cur, int supWdt, int nSupR,
                                int* nbpivot, double threshold, double* D_block);

    static void eigen_dtrsm2(const double* T, int rowNo, int supWdt,
                             double* B_ptr, int nSupR);

    static void row_reordering(int /*supNo*/, size_t* /*lC*/, int* /*Li_ptr*/,
                               int* /*lR*/, int* /*blockSet*/, int* /*atree_sm*/,
                               int* /*cT*/, int* /*rT*/, int* /*col2Sup*/,
                               double* /*lValues*/, std::vector<int> /*perm_req*/,
                               int* /*swap_full*/, int* /*xi*/,
                               int* /*map*/, int* /*ws*/, double* /*contribs*/);

//    static CSC* ptranspose(CSC* A,int /*dummy*/,const int* /*perm*/,
//                           void* /*workspace*/,int /*dosymm*/,int &status);


/*
 * LDLT factorization without pivoting or static pivoting
 */


bool ldl_left_sn_01 (int n, int* c, int* r, double* values,
                         size_t *lC, int * lR, size_t * Li_ptr, double* lValues,
                         double *D,
                         int *blockSet, int supNo, double *timing,
#ifndef PRUNE
                    int *aTree, int *cT, int *rT, int *col2Sup,
#else
  int *prunePtr, int *pruneSet,
#endif
                    int super_max, int col_max, int &nbpivot,
                     double threshold=1e-13);

}
#endif //PROJECT_SERIAL_BLOCKED_LDL_H
