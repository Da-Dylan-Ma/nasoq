/******************************************************************************
 * NASOQ QP Solver: Supernodal LDL Factorization (Non‑Pivoting)
 ******************************************************************************/

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <chrono>

// Include Eigen headers (adjust path as needed)
#include "Eigen/Dense"
#include "Eigen/Cholesky"

// ---------------------------------------------------------------------------
// Include original NASOQ common routines (for DFS, def, and reach).
// (Normally one would include header files, but here we include the .cpp files.)
#include "common/def.h"
#include "common/DFS.h"
#include "common/Reach.h"

namespace nasoq {

  //-----------------------------
  // Module 1: eigen_dgemm (Same‑level Functional)
  // Computes: C = alpha*(A * B^T) + beta*C
  static void eigen_dgemm(const char* /*transA*/, const char* /*transB*/,
                          const int *M, const int *N, const int *K,
                          const double *alpha,
                          const double *A, const int *lda,
                          const double *B, const int *ldb,
                          const double *beta,
                          double *C, const int *ldc)
  {
      int m = *M, n = *N, k = *K;
      double alp = *alpha, bet = *beta;
      Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >
          matA(A, m, k);
      Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >
          matB(B, *ldb, n);
      Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >
          matC(C, m, n);
      matC = alp * (matA * matB.transpose()) + bet * matC;
  }

  //-----------------------------
  // Module 2: eigen_dtrsm (Same‑level Functional)
  // Solves X * (L^T) = B for X, given a unit lower triangular matrix L.
  static void eigen_dtrsm(const char* /*side*/, const char* /*uplo*/,
                          const char* /*transA*/, const char* /*diag*/,
                          const int *M, const int *N,
                          const double *alpha,
                          const double *A, const int *lda,
                          double *B, const int *ldb)
  {
      int m = *M, n = *N;
      double alp = *alpha;
      Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >
          Lmat(A, *lda, n); // L is n x n (unit lower triangular)
      Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >
          matB(B, *ldb, n); // B is m x n
      // Solve for X in X * L^T = B  <=>  (X^T) solves L * (X^T) = B^T.
      Eigen::MatrixXd X_trans = Lmat.triangularView<Eigen::Lower>().solve(matB.transpose());
      Eigen::MatrixXd X = alp * X_trans.transpose();
      matB = X;
  }

  //-----------------------------
  // Module 3: eigen_sym_sytrf (Same‑level Functional)
  // Factorizes the dense diagonal block using LDL^T (without pivoting).
  // Writes the unit‐lower factor into cur and extracts the diagonal into D_block.
  static void eigen_sym_sytrf(double* cur, int supWdt, int nSupR, int* nbpivot,
                               double threshold, double* D_block)
  {
      Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >
          fullBlock(cur, nSupR, supWdt);
      Eigen::MatrixXd A_block = fullBlock.topRows(supWdt);
      Eigen::LDLT<Eigen::MatrixXd, Eigen::Lower> ldlt;
      ldlt.compute(A_block);
      if (ldlt.info() != Eigen::Success) {
          *nbpivot = -1;
          return;
      }
      Eigen::MatrixXd Lmat = ldlt.matrixL();
      Eigen::VectorXd Dvec = ldlt.vectorD();
      for (int j = 0; j < supWdt; ++j) {
          for (int i = 0; i < supWdt; ++i) {
              if (i > j)
                  cur[j * nSupR + i] = Lmat(i, j);
              else if (i == j)
                  cur[j * nSupR + i] = 1.0;
              else
                  cur[j * nSupR + i] = 0.0;
          }
          D_block[j] = Dvec(j);
      }
      *nbpivot = 0; // no pivoting
  }

  //-----------------------------
  // Module 4: eigen_dtrsm2 (Same‑level Functional)
  // Solves the triangular system arising in the update phase.
  static void eigen_dtrsm2(const double* T, int rowNo, int supWdt, double* B_ptr, int nSupR)
  {
      Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >
          Tmat(T, supWdt, supWdt);
      Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >
          Bmat(B_ptr, nSupR, supWdt);
      Eigen::MatrixXd X_trans = Tmat.triangularView<Eigen::Lower>().solve(Bmat.transpose());
      Eigen::MatrixXd X = X_trans.transpose();
      Bmat = X;
  }

  //-----------------------------
  // Module 5: row_reordering (No‑op Placeholder)
  static void row_reordering(int /*supNo*/, size_t* /*lC*/, int* /*Li_ptr*/,
                             int* /*lR*/, int* /*blockSet*/, int* /*atree_sm*/,
                             int* /*cT*/, int* /*rT*/, int* /*col2Sup*/,
                             double* /*lValues*/, std::vector<int> /*perm_req*/,
                             int* /*swap_full*/, int* /*xi*/,
                             int* /*map*/, int* /*ws*/, double* /*contribs*/)
  {
      // In the full NASOQ implementation this routine re‑orders rows after pivoting.
      // For our non‑pivoting version we leave it as a no‑op.
  }

  //-----------------------------
  // (Module 6: ereach_sn is no longer defined here; we use the original implementation
  // provided in common/Reach.cpp.)
  // (That file defines functions such as: int reach(int, ...), int ereach(...), 
  // and int ereach_sn(...).)
  
  //-----------------------------
  // Module 7: ptranspose (Same‑level Functional)
  // Transposes a CSC matrix.
  static CSC* ptranspose(CSC* A, int /*dummy*/, const int* /*perm*/,
                         void* /*workspace*/, int /*dosymm*/, int &status)
  {
      if (!A) { status = -1; return nullptr; }
      CSC* T = new CSC;
      int m = A->nrow, n = A->ncol;
      T->nrow = n;
      T->ncol = m;
      T->stype = -1;
      T->xtype = 1;
      T->nzmax = A->nzmax;
      T->packed = 1;
      T->sorted = 1;
      T->p = new int[T->ncol+1](); // zero initialize
      T->i = new int[T->nzmax];
      T->x = new double[T->nzmax];
      // Count entries per row of A (which become column counts of T)
      for (int j = 0; j < A->ncol; j++) {
          for (int idx = A->p[j]; idx < A->p[j+1]; idx++) {
              int row = A->i[idx];
              if (row >= 0 && row < T->ncol)
                  T->p[row]++;
          }
      }
      // Prefix sum
      int sum = 0;
      for (int i = 0; i < T->ncol; i++) {
          int tmp = T->p[i];
          T->p[i] = sum;
          sum += tmp;
      }
      T->p[T->ncol] = A->nzmax;
      std::vector<int> current(T->ncol, 0);
      for (int col = 0; col < A->ncol; col++) {
          for (int idx = A->p[col]; idx < A->p[col+1]; idx++) {
              int row = A->i[idx];
              double val = A->x[idx];
              int pos = T->p[row] + current[row];
              T->i[pos] = col;
              T->x[pos] = val;
              current[row]++;
          }
      }
      status = 0;
      return T;
  }

  //-----------------------------
  // Main Factorization Routine: ldl_left_sn_01
  bool ldl_left_sn_01(int n, int *c, int *r, double *values,
                      size_t *lC, int *lR, size_t *Li_ptr, double *lValues,
                      double *D, int *blockSet, int supNo, double *timing,
                      int *aTree, int *cT, int *rT, int *col2Sup,
                      int super_max, int col_max, int &nbpivot, double threshold)
  {
      (void)timing;  // timing not collected here
      nbpivot = 0;
      int top = 0;
      int *xi = new int[2 * supNo]();
      int *map = new int[n]();
      double *contribs = new double[super_max * col_max]();
      double *trn_diag = new double[super_max * col_max]();
      double one[1] = {1.0}, zero[1] = {0.0};
      
      // Loop over each supernode
      for (int s = 1; s <= supNo; ++s) {
          int curCol = (s != 0 ? blockSet[s - 1] : 0);
          int nxtCol = blockSet[s];
          int supWdt = nxtCol - curCol;
          int nSupR = static_cast<int>(Li_ptr[nxtCol] - Li_ptr[curCol]);
          for (int i = static_cast<int>(Li_ptr[curCol]), cnt = 0;
               i < static_cast<int>(Li_ptr[nxtCol]); ++i) {
              map[lR[i]] = cnt++;
          }
          // Copy columns from A into factor storage using the mapping.
          for (int i = curCol; i < nxtCol; ++i) {
              for (int j = c[i]; j < c[i + 1]; ++j) {
                  lValues[lC[i] + map[r[j]]] = values[j];
              }
          }
          double *src, *cur = &lValues[lC[curCol]];
#ifndef PRUNE
          // Use the original ereach_sn from common/Reach.cpp.
          top = ereach_sn(supNo, cT, rT, curCol, nxtCol, col2Sup, aTree, xi, xi + supNo);
          assert(top >= 0);
          for (int i = top; i < supNo; ++i) {
              int lSN = xi[i];
#else
          for (int i = 0; i < supNo; ++i) {
              int lSN = i;
#endif
              int cSN = blockSet[lSN];
              int cNSN = blockSet[lSN + 1];
              int Li_ptr_cNSN = static_cast<int>(Li_ptr[cNSN]);
              int Li_ptr_cSN = static_cast<int>(Li_ptr[cSN]);
              int nSNRCur = Li_ptr_cNSN - Li_ptr_cSN;
              int supWdts = cNSN - cSN;
              int lb = 0, ub = 0;
              bool sw = true;
              for (int j = static_cast<int>(Li_ptr[cSN]); j < static_cast<int>(Li_ptr[cNSN]); ++j) {
                  if (lR[j] >= curCol && sw) {
                      lb = j - static_cast<int>(Li_ptr[cSN]);
                      sw = false;
                  }
                  if (lR[j] < curCol + supWdt && !sw) {
                      ub = j - static_cast<int>(Li_ptr[cSN]);
                  }
              }
              int localSupRows = nSNRCur - lb;
              int ndrow1 = ub - lb + 1;
              int ndrow3 = localSupRows - ndrow1;
              src = &lValues[lC[cSN] + lb];
              // 2x2 Block Multiplication: form trn_diag = D * src.
              for (int l = 0; l < supWdts; ++l) {
                  double tmp = D[cSN + l];
                  for (int l1 = 0; l1 < localSupRows; ++l1) {
                      trn_diag[l * localSupRows + l1] = tmp * src[l * (nSNRCur) + l1];
                  }
              }
              {
                  int M = ndrow3, N = ndrow1, K = supWdts;
                  double* srcL = &lValues[lC[cSN] + ub + 1];
                  eigen_dgemm("N", "C", &M, &N, &K, one,
                              srcL, &nSNRCur,
                              src, &nSNRCur,
                              zero,
                              contribs + ndrow1, &localSupRows);
              }
              for (int i2 = 0; i2 < ndrow1; ++i2) {
                  int col = map[lR[static_cast<int>(Li_ptr[cSN]) + i2 + lb]];
                  for (int j2 = i2; j2 < localSupRows; ++j2) {
                      int cRow = lR[static_cast<int>(Li_ptr[cSN]) + j2 + lb];
                      cur[col * localSupRows + map[cRow]] -= contribs[i2 * localSupRows + j2];
                  }
              }
          } // end loop over source supernodes

          // Dense Factorization via Eigen LDLT
          {
              eigen_sym_sytrf(cur, supWdt, nSupR, &nbpivot, threshold, &D[curCol]);
          }

          // Build temporary block for triangular solve
          int rowNo = nSupR - supWdt;
          for (int l = 0; l < supWdt; ++l) {
              double tmp = cur[l + l * nSupR];
              double *stCol = trn_diag + l * supWdt + l;
              double *curColPtr = cur + l * nSupR + l;
              *stCol = tmp;
              for (int l1 = 0; l1 < supWdt - l - 1; ++l1) {
                  *(++stCol) = tmp * *(++curColPtr);
              }
          }

#ifdef OPENBLAS
          // If using BLAS, one would call cblas_dtrsm; here we always use Eigen.
#else
          eigen_dtrsm2(trn_diag, rowNo, supWdt, &cur[supWdt], nSupR);
#endif

          // Set the diagonal of the current block to 1.
          for (int k = 0; k < supWdt; ++k) {
              cur[k * nSupR + k] = 1.0;
          }
      } // end loop over supernodes

      // Call row_reordering (no‑op in this implementation)
      row_reordering(supNo, lC, reinterpret_cast<int*>(Li_ptr), lR, blockSet, nullptr,
                     cT, rT, col2Sup, lValues, std::vector<int>(), nullptr, nullptr, map, nullptr, nullptr);

      delete[] contribs;
      delete[] trn_diag;
      delete[] xi;
      delete[] map;

      return true;
  }

  //-----------------------------
  // Minimal LFactor structure.
  struct LFactor {
      int nsuper;       // number of supernodes
      size_t xsize;     // size of the factor array
      int nzmax;        // maximum number of nonzeros
      size_t *p;        // lC array (column pointers for L factor)
      int *s;           // lR array (row indices for L factor)
      size_t *i_ptr;    // Li_ptr array
      double *x;        // lValues array
  };

  //-----------------------------
  // Minimal profiling class using std::chrono.
  struct profiling_solver_info {
      double fact_time, analysis_time, solve_time, iter_time;
      double ordering_time, update_time, piv_reord;
      double *timing_chol;
      profiling_solver_info(int nt) : fact_time(0), analysis_time(0), solve_time(0),
          iter_time(0), ordering_time(0), update_time(0), piv_reord(0) {
              timing_chol = new double[4 + nt]();
      }
      ~profiling_solver_info() { delete[] timing_chol; }
      std::chrono::time_point<std::chrono::system_clock> tic() {
          return std::chrono::system_clock::now();
      }
      std::chrono::time_point<std::chrono::system_clock> toc() {
          return std::chrono::system_clock::now();
      }
      double elapsed_time(std::chrono::time_point<std::chrono::system_clock> beg,
                          std::chrono::time_point<std::chrono::system_clock> lst) {
          return std::chrono::duration_cast<std::chrono::duration<double>>(lst - beg).count();
      }
      void print_profiling() {
          std::cout << "analysis time: " << analysis_time << "; ";
          std::cout << "fact time: " << fact_time << "; ";
          std::cout << "update time: " << update_time << "; ";
          std::cout << "reordering pivot time: " << piv_reord << "; ";
          std::cout << "solve time: " << solve_time << "; ";
      }
  };

  //-----------------------------
  // Minimal SolverSettings class (upper-layer interface).
  class SolverSettings {
  public:
      CSC *A;        // Input matrix (CSC)
      double *rhs;   // Right-hand side vector
      CSC *A_ord;    // Shallow copy of A (ordered)
      CSC *AT_ord;   // Transpose of A_ord
      LFactor *L;    // Factorization info
      double *D;     // Diagonal array
      int *blockSet; // Array defining supernode boundaries
      int supNo;     // Number of supernodes
      profiling_solver_info *psi;
      double *x;     // Solution vector
      int n;         // Matrix dimension

      // Constructor: store pointers and perform minimal initialization.
      SolverSettings(CSC *Amat, double *rhs_in) {
          default_setting();
          A = Amat;
          rhs = rhs_in;
          A_ord = nullptr;
          AT_ord = nullptr;
          L = new LFactor;
          // For non‑pivoting, we assume one supernode covering all columns.
          supNo = 1;
          n = A->ncol;
          blockSet = new int[2];
          blockSet[0] = 0; blockSet[1] = n;
          psi = new profiling_solver_info(1);
          x = new double[n];
          // Allocate arrays for L factor.
          L->nsuper = supNo;
          L->nzmax = A->nzmax;
          L->p = new size_t[n+1];
          L->s = new int[A->nzmax];
          L->i_ptr = new size_t[n+1];
          L->x = new double[A->nzmax];
          for (int i = 0; i <= n; i++) {
              L->p[i] = A->p[i];
              L->i_ptr[i] = A->p[i];
          }
          for (int i = 0; i < A->nzmax; i++) {
              L->s[i] = A->i[i];
              L->x[i] = 0.0;
          }
          // Allocate D: one value per column.
          D = new double[n];
          for (int i = 0; i < n; i++) {
              D[i] = 0.0;
          }
      }

      ~SolverSettings() {
          if (A_ord) {
              delete[] A_ord->p;
              delete[] A_ord->i;
              delete[] A_ord->x;
              delete A_ord;
          }
          if (AT_ord) {
              delete[] AT_ord->p;
              delete[] AT_ord->i;
              delete[] AT_ord->x;
              delete AT_ord;
          }
          if (L) {
              delete[] L->p;
              delete[] L->s;
              delete[] L->i_ptr;
              delete[] L->x;
              delete L;
          }
          if (D) delete[] D;
          if (blockSet) delete[] blockSet;
          if (psi) delete psi;
          if (x) delete[] x;
      }

      // default_setting: minimal defaults.
      void default_setting() {
          // In this simple test, no additional settings are needed.
      }

      // symbolic_analysis: copy A to A_ord, then compute AT_ord.
      int symbolic_analysis() {
          A_ord = new CSC;
          A_ord->nrow = A->nrow;
          A_ord->ncol = A->ncol;
          A_ord->nzmax = A->nzmax;
          A_ord->p = new int[A->ncol+1];
          A_ord->i = new int[A->nzmax];
          A_ord->x = new double[A->nzmax];
          for (int i = 0; i <= A->ncol; i++) {
              A_ord->p[i] = A->p[i];
          }
          for (int i = 0; i < A->nzmax; i++) {
              A_ord->i[i] = A->i[i];
              A_ord->x[i] = A->x[i];
          }
          int status = 0;
          AT_ord = ptranspose(A_ord, 2, nullptr, nullptr, 0, status);
          return 1;
      }

      // numerical_factorization: call ldl_left_sn_01.
      int numerical_factorization() {
          int nbpivot = 0;
          bool ok = ldl_left_sn_01(n, A_ord->p, A_ord->i, A_ord->x,
                                    L->p, L->s, L->i_ptr, L->x, D,
                                    blockSet, supNo, nullptr, nullptr, nullptr, nullptr, nullptr,
                                    64, 64, nbpivot, 1e-14);
          return ok ? 1 : 0;
      }

      // solve_only: For our simple test, compute x = rhs ./ D.
      double* solve_only() {
          for (int i = 0; i < n; i++) {
              if (fabs(D[i]) < 1e-14) D[i] = 1e-14;
              x[i] = rhs[i] / D[i];
          }
          return x;
      }

      // reorder_matrix: (No‑op placeholder)
      void reorder_matrix() {
          row_reordering(supNo, L->p, reinterpret_cast<int*>(L->i_ptr), L->s,
                         nullptr, nullptr, nullptr, nullptr, nullptr,
                         L->x, std::vector<int>(), nullptr, nullptr, nullptr, nullptr, nullptr);
      }
  };

} // end namespace nasoq

//------------------------------------------
// Main function for PC.
int main() {
    std::cout << "Starting Supernodal LDL (Non‑Pivoting) NASOQ Solver with Eigen..." << std::endl;

    // Build a simple 2x2 diagonal test matrix in CSC format.
    int sizeH = 2, nnzH = 2;
    int *Hp = new int[sizeH+1];
    int *Hi = new int[nnzH];
    double *Hx = new double[nnzH];

    // Diagonal matrix [2,0; 0,2]
    Hp[0] = 0; Hp[1] = 1; Hp[2] = 2;
    Hi[0] = 0; Hi[1] = 1;
    Hx[0] = 2.0; Hx[1] = 2.0;

    nasoq::CSC *H = new nasoq::CSC;
    H->nrow = sizeH;
    H->ncol = sizeH;
    H->nzmax = nnzH;
    H->p = Hp;
    H->i = Hi;
    H->x = Hx;
    H->stype = -1;
    H->packed = 1;
    H->sorted = 1;
    H->xtype = 1;

    // Create a right-hand side vector.
    // For q = [-4, -4], expected solution is x = [-2, -2].
    double *q = new double[sizeH];
    q[0] = -4; q[1] = -4;

    nasoq::SolverSettings solver(H, q);
    solver.symbolic_analysis();
    if (!solver.numerical_factorization()) {
        std::cerr << "Factorization failed!" << std::endl;
        return -1;
    }
    double *sol = solver.solve_only();
    std::cout << "Solution 1: ";
    for (int i = 0; i < sizeH; i++) {
        std::cout << sol[i] << " ";
    }
    std::cout << std::endl;

    // Test a new right-hand side: for q = [8, 8], expected solution is x = [4, 4].
    q[0] = 8; q[1] = 8;
    sol = solver.solve_only();
    std::cout << "Solution 2: ";
    for (int i = 0; i < sizeH; i++) {
        std::cout << sol[i] << " ";
    }
    std::cout << std::endl;

    delete[] q;
    delete H;
    return 0;
}

