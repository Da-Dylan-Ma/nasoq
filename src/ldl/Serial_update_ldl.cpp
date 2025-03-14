//
// Created by Shujian Qian on 2020-10-29.
//


#include <cassert>
#include <chrono>
#include <cmath>

#include "nasoq/ldl/Serial_update_ldl.h"

#include "nasoq/common/Reach.h"
#include "nasoq/common/Sym_BLAS.h"
#include <Eigen/Dense>

namespace nasoq {

    void custom_sym_dgemm(const char* transA, const char* transB,
                          const int* M, const int* N, const int* K,
                          const double* alpha,
                          const double* A, const int* ldA,
                          const double* B, const int* ldB,
                          const double* beta,
                          double* C, const int* ldC)
    {
        bool A_is_trans = (transA[0] == 'T' || transA[0] == 't' || transA[0] == 'C' || transA[0] == 'c');
        bool B_is_trans = (transB[0] == 'T' || transB[0] == 't' || transB[0] == 'C' || transB[0] == 'c');

        Eigen::Map<
                const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>,
                0,
                Eigen::OuterStride<>
        > matA(
                A,
                (A_is_trans ? *K : *M),
                (A_is_trans ? *M : *K),
                Eigen::OuterStride<>(*ldA)
        );

        Eigen::Map<
                const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>,
                0,
                Eigen::OuterStride<>
        > matB(
                B,
                (B_is_trans ? *N : *K),
                (B_is_trans ? *K : *N),
                Eigen::OuterStride<>(*ldB)
        );

        Eigen::Map<
                Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>,
                0,
                Eigen::OuterStride<>
        > matC(
                C,
                *M,
                *N,
                Eigen::OuterStride<>(*ldC)
        );

        Eigen::MatrixXd Aop;
        if (A_is_trans) {
            Aop = matA.transpose();
        } else {
            Aop = matA;
        }

        Eigen::MatrixXd Bop;
        if (B_is_trans) {
            Bop = matB.transpose();
        } else {
            Bop = matB;
        }

        matC = (*beta) * matC + (*alpha) * (Aop * Bop);
    }

    // TODO: Double check this, leading to weird intermediate numbers
    int custom_dsytrf(int matrix_layout, char uplo, int n, double* a, int lda, int* ipiv) {
        if (matrix_layout != 101) return -1;
        if (uplo != 'L' && uplo != 'l') return -2;
        if (n <= 0) return 0;
        typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatC;
        Eigen::Map<MatC, 0, Eigen::OuterStride<>> M(a, n, n, Eigen::OuterStride<>(lda));
        double alpha = (1.0 + std::sqrt(17.0)) / 8.0;
        int info = 0;
        int k = n;
        while (k > 0) {
            int kstep = 1;
            int imax = k;
            double absakk = std::fabs(M(k - 1, k - 1));
            double colmax = 0.0;
            if (k > 1) {
                for (int i = 0; i < k - 1; i++) {
                    double val = std::fabs(M(k - 1, i));
                    if (val > colmax) {
                        colmax = val;
                        imax = i + 1;
                    }
                }
                if (absakk < alpha * colmax) {
                    double rowmax = 0.0;
                    int irow = imax - 1;
                    for (int j = k - 1; j < n; j++) {
                        double val = std::fabs(M(irow, j));
                        if (val > rowmax) rowmax = val;
                    }
                    if (std::fabs(M(irow, irow)) < alpha * rowmax) kstep = 2;
                }
            }
            if (kstep == 1) {
                if (imax != k) {
                    M.row(k - 1).swap(M.row(imax - 1));
                    M.col(k - 1).swap(M.col(imax - 1));
                    ipiv[k - 1] = imax;
                } else {
                    ipiv[k - 1] = k;
                }
                if (std::fabs(M(k - 1, k - 1)) < 1e-15 && info == 0) info = k;
                if (k > 1) {
                    double pivot = M(k - 1, k - 1);
                    if (std::fabs(pivot) < 1e-15) pivot = (pivot < 0.0 ? -1e-15 : 1e-15);
                    for (int i = 0; i < k - 1; i++) M(k - 1, i) /= pivot;
                    for (int j = 0; j < k - 1; j++) {
                        double c = M(k - 1, j);
                        for (int i = 0; i <= j; i++) M(j, i) -= c * M(k - 1, i);
                    }
                }
                k--;
            } else {
                if (k < 2) {
                    ipiv[k - 1] = k;
                    if (std::fabs(M(k - 1, k - 1)) < 1e-15 && info == 0) info = k;
                    k--;
                    continue;
                }
                int km1 = k - 1;
                if (imax != km1) {
                    M.row(km1 - 1).swap(M.row(imax - 1));
                    M.col(km1 - 1).swap(M.col(imax - 1));
                }
                double d12 = M(k - 1, km1 - 1);
                double d11 = M(km1 - 1, km1 - 1);
                double d22 = M(k - 1, k - 1);
                double det = d11 * d22 - d12 * d12;
                if (std::fabs(det) < 1e-15 && info == 0) info = k;
                M(km1 - 1, km1 - 1) = d22 / det;
                M(k - 1, k - 1) = d11 / det;
                M(k - 1, km1 - 1) = -d12 / det;
                ipiv[k - 1] = -imax;
                ipiv[k - 2] = -imax;
                if (k > 2) {
                    for (int i = 0; i < k - 2; i++) {
                        double akm1 = M(km1 - 1, i);
                        double ak = M(k - 1, i);
                        M(km1 - 1, i) = (d22 * akm1 - d12 * ak) / det;
                        M(k - 1, i) = (-d12 * akm1 + d11 * ak) / det;
                    }
                    for (int j = 0; j < k - 2; j++) {
                        double v1 = M(km1 - 1, j);
                        double v2 = M(k - 1, j);
                        for (int i = j; i < k - 2; i++) {
                            M(i, j) -= (v1 * M(km1 - 1, i) + v2 * M(k - 1, i));
                        }
                    }
                }
                k -= 2;
            }
        }
        return info;
    }

    // TODO: Double check this, leading to a higher lag-res
    int custom_dlapmt(int matrix_layout, int forwrd, int m, int n, double* x, int ldx, int* k) {
        if (matrix_layout != 101) return -1;
        if (m <= 0 || n <= 0) return 0;

        Eigen::Map<
                Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>,
                0,
                Eigen::OuterStride<>
        > M(x, m, n, Eigen::OuterStride<>(ldx));

        Eigen::MatrixXd tmp(m, n);

        // Forward permutation: new col J = old col K(J).
        // "X(*,K(J)) is moved to X(*,J) for J=1..N."
        if (forwrd) {
            for (int j = 0; j < n; j++) {
                int oldCol = k[j] - 1;
                tmp.col(j) = M.col(oldCol);
            }
        }
            // Backward permutation: new col K(J) = old col J.
            // "X(*,J) is moved to X(*,K(J)) for J=1..N."
        else {
            for (int j = 0; j < n; j++) {
                int newCol = k[j] - 1;
                tmp.col(newCol) = M.col(j);
            }
        }

        M = tmp;
        return 0;
    }

    // TODO: Double check this
    void custom_dtrsm(const char* side,
                          const char* uplo,
                          const char* trans,
                          const char* diag,
                          const int* M,
                          const int* N,
                          const double* alpha,
                          const double* A,
                          const int* lda,
                          double* B,
                          const int* ldb)
    {
        bool sideR = (side[0] == 'R' || side[0] == 'r');
        bool uploL = (uplo[0] == 'L' || uplo[0] == 'l');
        bool transT = (trans[0] == 'T' || trans[0] == 't' || trans[0] == 'C' || trans[0] == 'c');
        bool unitD = (diag[0] == 'U' || diag[0] == 'u');
        if (!sideR || !uploL || !transT || !unitD) {
            Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>,
                    0, Eigen::OuterStride<> >
                    matB(B, *M, *N, Eigen::OuterStride<>(*ldb));
            if (*alpha != 1.0) {
                matB *= (*alpha);
            }
            return;
        }
        if (*N <= 0 || *M < 0) return;
        if (*lda < *N) return;
        Eigen::MatrixXd Ablock = Eigen::MatrixXd::Zero(*N, *N);
        for (int col = 0; col < *N; col++) {
            for (int row = col; row < *N; row++) {
                Ablock(row, col) = A[col*(*lda) + row];
            }
        }
        for (int i = 0; i < *N; i++) {
            Ablock(i,i) = 1.0;
        }
        Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>,
                0, Eigen::OuterStride<> >
                matB(B, *M, *N, Eigen::OuterStride<>(*ldb));
        if (*alpha != 1.0) {
            matB *= (*alpha);
        }
        matB.transposeInPlace();
        Eigen::MatrixXd Aupper = Ablock.transpose();
        Aupper.triangularView<Eigen::Upper>().solveInPlace(matB);
        matB.transposeInPlace();
    }


 bool
 update_ldl_left_sn_02_v2(int n, int *c, int *r, double *values, size_t *lC, int *lR, size_t *Li_ptr, double *lValues,
                          double *D, int *blockSet, int supNo, double *timing, int *aTree, int *cT, int *rT,
                          int *col2Sup, std::vector<int> mod_indices, int super_max, int col_max, int &nbpivot,
                          int *perm_piv, int *atree_sm, int *ws_int, double *ws_dbl, double threshold) {
#if defined(OPENBLAS) && defined(NASOQ_USE_CLAPACK)
  using nasoq::clapacke::LAPACKE_dlapmt;
  using nasoq::clapacke::LAPACKE_dsytrf;
#endif
  /*
   * For timing using BLAS
   */
  std::chrono::time_point<std::chrono::system_clock> start, end1;
  double duration1 = 0;
  const int incx = 1;
  int top = 0;
  int *xi;
  int *swap_full;
  int *map;
  int *ws;
  int *ipiv;

  double *contribs;
  double *trn_diag;

  if (ws_int != NULL) {
   int ws_size = 4 * super_max + 2 * n + 3 * supNo;
   std::fill_n(ws_int, ws_size, 0);
   xi = ws_int;
   swap_full = ws_int + 3 * supNo;
   map = ws_int + 3 * supNo + n;
   ws = ws_int + 3 * supNo + 2 * n;
   ipiv = ws_int + 3 * supNo + 2 * n + 3 * (super_max);
  } else {
   xi = new int[3 * supNo]();
   swap_full = new int[n]();
   map = new int[n]();
   ws = new int[3 * super_max];
   ipiv = new int[super_max]();
  }

  if (ws_dbl != NULL) {
   int ws_dbl_size = 2 * super_max * col_max;
   std::fill_n(ws_dbl, ws_dbl_size, 0.0);
   contribs = ws_dbl;
   trn_diag = ws_dbl + super_max * col_max;
  } else {
   contribs = new double[super_max * col_max]();
   trn_diag = new double[super_max * col_max]();
  }

  int info;
  std::vector<int> perm_req;
  double one[2], zero[2];
  one[0] = 1.0;    /* ALPHA for *syrk, *herk, *gemm, and *trsm */
  one[1] = 0.;
  zero[0] = 0.;     /* BETA for *syrk, *herk, and *gemm */
  zero[1] = 0.;
  for (int i = 0; i < n; ++i) {
   perm_piv[i] = i;
  }
  //for (int s = 1; s <= supNo; ++s) {
  for (int mn = 0; mn < mod_indices.size(); ++mn) {
   int s = mod_indices[mn] + 1;
   int curCol = s != 0 ? blockSet[s - 1] : 0;
   int nxtCol = blockSet[s];
   int supWdt = nxtCol - curCol;
   int nSupR = Li_ptr[nxtCol] - Li_ptr[curCol];//row size of supernode
   for (int i = Li_ptr[curCol], cnt = 0; i < Li_ptr[nxtCol]; ++i) {
    map[lR[i]] = cnt++;//mapping L rows position to actual row idx
   }
   double *src, *cur = &lValues[lC[curCol]], *cur_d = &D[curCol];//pointing to first element of the current supernode
   // Reseting the current supernode.
   for (int i = 0; i < supWdt; ++i) {
    cur_d[i] = 0;
    cur_d[i + n] = 0;
    for (int j = 0; j < nSupR; ++j) {
     cur[i * nSupR + j] = 0.0;
    }
   }

   //copy the columns from A to L
   for (int i = curCol; i < nxtCol; ++i) {//Copy A to L
    int pad = i - curCol;
    for (int j = c[i]; j < c[i + 1]; ++j) {
     lValues[lC[i] + map[r[j]]] = values[j];
     //std::cout<<r[j]<<":"<<lC[i]+map[r[j]]<<";"<<values[j]<<";";
    }
    //  std::cout<<"\n";
   }
#if 0
   for (int i = curCol; i < nxtCol; ++i) {//Copy A to L
    std::cout<<"\n";
    for (int j = lC[i]; j < lC[i+1] ; ++j) {
     std::cout<<lValues[j]<<";";
    }
   }
   std::cout<<"\n\n";
#endif

   top = ereach_sn(supNo, cT, rT, curCol, nxtCol, col2Sup, aTree, xi, xi + supNo);
   assert(top >= 0);
   //int *lbs = xi+supNo, *ubs = xi + 2*supNo;//To use for row permutation
   //if(s==2){top =2; xi[top] = 0;}
   //std::cout<<"**: "<<supNo-top<<"\n";
   for (int i = top; i < supNo; ++i) {
    int lSN = xi[i];
    int nSupRs = 0;

    int cSN = blockSet[lSN];//first col of current SN
    int cNSN = blockSet[lSN + 1];//first col of Next SN
    int Li_ptr_cNSN = Li_ptr[cNSN];
    int Li_ptr_cSN = Li_ptr[cSN];
    int nSNRCur = Li_ptr_cNSN - Li_ptr_cSN;
    int supWdts = cNSN - cSN;//The width of current src SN
    int lb = 0, ub = 0;
    bool sw = true;
    int beg_col = cSN, end_col = 0;
    for (int j = Li_ptr_cSN; j < Li_ptr_cNSN; ++j) {
     //finding the overlap between curCol and curCol+supWdt in the src col
     if (lR[j] >= curCol && sw) {
      //src*transpose(row lR[j])
      lb = j - Li_ptr_cSN;
      //lbs[i] = lb;
      sw = false;
     }
     if (lR[j] < curCol + supWdt && !sw) {
      ub = j - Li_ptr_cSN;
      //ubs[i] = ub;
     }
    }
    nSupRs = Li_ptr_cNSN - Li_ptr_cSN - lb;
    int ndrow1 = ub - lb + 1;
    int ndrow3 = nSupRs - ndrow1;
    src = &lValues[lC[cSN] + lb];//first element of src supernode starting from row lb
    double *srcL = &lValues[lC[cSN] + ub + 1];

    // TODO: Replace this
    blocked_2by2_mult(supWdts, nSupRs, &D[cSN], src, trn_diag, nSNRCur, n);

    custom_sym_dgemm("N", "C", &nSupRs, &ndrow1, &supWdts, one, trn_diag, &nSupRs,
          src, &nSNRCur, zero, contribs, &nSupRs);

       //copying contrib to L
    for (int i = 0; i < ndrow1; ++i) {//Copy contribs to L
     int col = map[lR[Li_ptr_cSN + i + lb]];//col in the SN
     //double ddiag = 1.0 ;/// D[col];
     for (int j = i; j < nSupRs; ++j) {
      int cRow = lR[Li_ptr_cSN + j + lb];//corresponding row in SN
      //lValues[lC[curCol+col]+ map[cRow]] -= contribs[i*nSupRs+j];
      cur[col * nSupR + map[cRow]] -= contribs[i * nSupRs + j];
//     if ( true){
//      std::cout<<"\n====="<<cSN<<"|| "<< cRow<<";;"<<contribs[i*nSupRs+j]<<";;"
//               <<cur[col*nSupR+map[cRow]]<<";;"<<"\n";
//     }
     }
    }
   }

   // TODO: Replace this
   LAPACKE_dsytrf(LAPACK_COL_MAJOR, 'L', supWdt, cur, nSupR, ipiv);
   // custom_dsytrf(LAPACK_COL_MAJOR, 'L', supWdt, cur, nSupR, ipiv);
   // TODO: Replace this, this looks fine but it is in the sysblas file
   int is_perm = reorder_after_sytrf(supWdt, cur, nSupR, ipiv,
                                     &perm_piv[curCol], &D[curCol], n, &swap_full[curCol], ws + supWdt);
   // re-order the columns of the super-node
   int rowNo = nSupR - supWdt;
   for (int m = 0; m < supWdt; ++m) {
    perm_piv[curCol + m]++;
   }

   if (is_perm) {
       // TODO: Replace this
//    LAPACKE_dlapmt(LAPACK_COL_MAJOR, 1, rowNo, supWdt, &cur[supWdt], nSupR, &perm_piv[curCol]);
     custom_dlapmt(LAPACK_COL_MAJOR, 1, rowNo, supWdt, &cur[supWdt], nSupR, &perm_piv[curCol]);
    perm_req.push_back(s);
   }

   //reordering row
   for (int k1 = 0; k1 < supWdt; ++k1) {
    perm_piv[curCol + k1] += (curCol - 1);
    // perm_piv++;
   }
   for (int l = 0; l < supWdt; ++l) {
    D[curCol + l] = cur[l + l * nSupR];
    cur[l + l * nSupR] = 1.0;
   }
//  /////
//  std::cout<<"\n";
//  for (int i = 0; i < supWdt; ++i) {
//   std::cout<<D[curCol]<<";";
//   for (int j = 0; j < nSupR ; ++j) {
//    std::cout<<cur[i*nSupR+j]<<";";
//   }
//   std::cout<<"\n";
//  }
//  std::cout<<"\n";
//  /////

#ifdef OPENBLAS
   cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasConjTrans, CblasUnit, rowNo, supWdt, 1.0,
               cur, nSupR, &cur[supWdt], nSupR);
#else
      // TODO: Replace this
   SYM_DTRSM("R", "L", "C", "U", &rowNo, &supWdt, one,
         cur, &nSupR, &cur[supWdt], &nSupR);
#endif

//      custom_dtrsm("R", "L", "C", "U", &rowNo, &supWdt, one,
//                   cur, &nSupR, &cur[supWdt], &nSupR);


//  /////
//  std::cout<<"\n";
//  for (int i = 0; i < supWdt; ++i) {
//   std::cout<<D[curCol]<<";";
//   for (int j = 0; j < nSupR ; ++j) {
//    std::cout<<cur[i*nSupR+j]<<";";
//   }
//   std::cout<<"\n";
//  }
//  std::cout<<"\n";
//  /////

    // TODO: Replace this
   blocked_2by2_solver(supWdt, &D[curCol], &cur[supWdt], rowNo, nSupR, n);

/*  /////
  std::cout<<"\n";
  for (int i = 0; i < supWdt; ++i) {
   std::cout<<D[curCol]<<";";
   for (int j = 0; j < nSupR ; ++j) {
    std::cout<<cur[i*nSupR+j]<<";";
   }
   std::cout<<"\n";
  }
  std::cout<<"\n";
  /////*/

  }

/* std::chrono::time_point<std::chrono::system_clock> start, end1;
 start = std::chrono::system_clock::now();*/
  for (int k = 0; k < super_max; ++k) {
   ws[k] = 0;
  }
  row_reordering(supNo, lC, Li_ptr, lR, blockSet, atree_sm, cT, rT, col2Sup,
                 lValues, perm_req, swap_full, xi, map, ws, contribs);
/* end1 = std::chrono::system_clock::now();
 std::chrono::duration<double> elapsed_seconds = end1-start;
 double duration1=elapsed_seconds.count();
 std::cout<<"++ " <<duration1<<"\n";*/
  nbpivot = perm_req.size();
  if (ws_int == NULL) {
   delete[]xi;
   delete[]map;
   delete[]ws;
   delete[]swap_full;
   delete[]ipiv;
  }
  if (ws_dbl == NULL) {
   delete[]contribs;
   delete[]trn_diag;
  }

  return true;
 }
}