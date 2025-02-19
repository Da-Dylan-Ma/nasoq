//
// Created by kazem on 8/15/20.
//

#include <nasoq/QP/linear_solver_wrapper.h>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <cstring>
#include <vector>

#include "yaml_qp_parser.h"


void transpose_unsym(const nasoq::CSC *A, nasoq::CSC *&B) {
  if (!A) return;  // no-op if A is null

  int m = A->nrow;
  int n = A->ncol;

  // Allocate B as A^T
  B = new nasoq::CSC;
  B->nrow = n;
  B->ncol = m;
  B->stype = 0;        // unsymmetric
  B->xtype = A->xtype;
  B->sorted = 1;       // we will produce sorted columns
  B->packed = 1;
  B->nz = A->nz;
  B->nzmax = A->nzmax;

  // Allocate column pointer
  B->p = new int[B->ncol + 1];
  std::memset(B->p, 0, (B->ncol + 1) * sizeof(int));

  B->i = new int[B->nzmax];
  if (A->x) {
    B->x = new double[B->nzmax];
  } else {
    B->x = nullptr;
  }

  // ----------------------------------------------------------------------
  // 1. Count how many entries go into each column of B.
  //    (which is each row of A)
  // ----------------------------------------------------------------------
  for (int colA = 0; colA < n; colA++) {
    for (int pA = A->p[colA]; pA < A->p[colA + 1]; pA++) {
      int rowA = A->i[pA];
      B->p[rowA + 1]++;
    }
  }

  // ----------------------------------------------------------------------
  // 2. Prefix sum of B->p to get the correct column pointers
  //    B->p[i] will be the start of column i in B->i,B->x
  // ----------------------------------------------------------------------
  for (int colB = 0; colB < B->ncol; colB++) {
    B->p[colB + 1] += B->p[colB];
  }
  // Now B->p[k] = cumulative # of nonzeros up to column k.

  // ----------------------------------------------------------------------
  // 3. Fill B->i and B->x
  //    Use an auxiliary "next position" array that
  //    starts at B->p[colB] and moves forward.
  // ----------------------------------------------------------------------
  std::vector<int> nextPos(B->ncol);
  for (int colB = 0; colB < B->ncol; colB++) {
    nextPos[colB] = B->p[colB];
  }

  for (int colA = 0; colA < n; colA++) {
    for (int pA = A->p[colA]; pA < A->p[colA + 1]; pA++) {
      int rowA = A->i[pA];
      int destPos = nextPos[rowA]++;
      B->i[destPos] = colA;
      if (B->x) {
        B->x[destPos] = A->x[pA];
      }
    }
  }
}


static void print_qp_debug(const QPProblem &qp){
  // Hessian H
  if(qp.H){
    std::cout<<"H: "<<qp.H->nrow<<" x "<<qp.H->ncol
             <<" nnz="<<qp.H->nzmax<<"\n";
    int show_cols = std::min((int)qp.H->ncol, 5);
    for(int col=0; col< show_cols; col++){
      int start= qp.H->p[col];
      int end  = qp.H->p[col+1];
      for(int idx=start; idx<end; idx++){
        int row= qp.H->i[idx];
        if(row<5){
          double val= qp.H->x[idx];
          std::cout<<"  H("<<row<<","<<col<<")="<<val<<"\n";
        }
      }
    }
  }

  // linear q
  if(qp.q){
    std::cout<<"q: [";
    int show_q= std::min(qp.n,5);
    for(int i=0;i< show_q;i++){
      std::cout<< qp.q[i]<<" ";
    }
    std::cout<<( (qp.n>5) ? "...]\n" : "]\n");
  }

  // equality constraints A, b
  if(qp.A){
    std::cout<<"A: "<<qp.A->nrow<<" x "<<qp.A->ncol
             <<" nnz="<<qp.A->nzmax<<"\n";
    int show_cols = std::min((int)qp.A->ncol,5);
    for(int col=0; col< show_cols; col++){
      int start= qp.A->p[col];
      int end  = qp.A->p[col+1];
      for(int idx=start; idx<end; idx++){
        int row= qp.A->i[idx];
        if(row<5){
          double val= qp.A->x[idx];
          std::cout<<"  A("<<row<<","<<col<<")="<<val<<"\n";
        }
      }
    }
  }
  if(qp.b){
    std::cout<<"b: [";
    int show_b= std::min(qp.me,5);
    for(int i=0;i< show_b;i++){
      std::cout<< qp.b[i]<<" ";
    }
    std::cout<<( (qp.me>5) ? "...]\n" : "]\n");
  }

  // inequality constraints C, l, u
  if(qp.C){
    std::cout<<"C: "<<qp.C->nrow<<" x "<<qp.C->ncol
             <<" nnz="<<qp.C->nzmax<<"\n";
    int show_cols = std::min((int)qp.C->ncol,5);
    for(int col=0; col< show_cols; col++){
      int start= qp.C->p[col];
      int end  = qp.C->p[col+1];
      for(int idx=start; idx<end; idx++){
        int row= qp.C->i[idx];
        if(row<5){
          double val= qp.C->x[idx];
          std::cout<<"  C("<<row<<","<<col<<")="<<val<<"\n";
        }
      }
    }
  }
  if(qp.l && qp.u){
    int show_ineq= std::min(qp.mi,5);
    std::cout<<"l: [";
    for(int i=0;i< show_ineq;i++){
      std::cout<< qp.l[i]<<" ";
    }
    std::cout<<( (qp.mi>5) ? "...]\n" : "]\n");
    std::cout<<"u: [";
    for(int i=0;i< show_ineq;i++){
      std::cout<< qp.u[i]<<" ";
    }
    std::cout<<( (qp.mi>5) ? "...]\n" : "]\n");
  }
}


/*
 * Solving Hx = q
 * H is a sparse matrix stored in CSC format
 * q is a dense array
 */

int main(int argc, char *argv[]){
  if(argc<2){
    std::cout<<"usage: "<<argv[0]<<" <qp_smp.yml>\n";
    return 1;
  }
  std::string fname=argv[1];
  QPProblem qp;
  bool ok=parse_qp_yaml(fname,qp);
  if(!ok){
    std::cerr<<"parse failed\n";
    return 2;
  }
  std::cout<<"parsed n="<<qp.n<<", me="<<qp.me<<", mi="<<qp.mi<<"\n";

  print_qp_debug(qp);

  // 2) Transpose A
  nasoq::CSC *AT = nullptr;
  transpose_unsym(qp.A, AT);

  // 3) Transpose C
  nasoq::CSC *CT = nullptr;
  transpose_unsym(qp.C, CT);

 /// Solving the linear system
 auto *lbl = new nasoq::SolverSettings(qp.H, qp.q, qp.A, AT, qp.C, CT);
 lbl->ldl_variant = 2; // set it to 4 if your C++ compiler supports openmp
 lbl->req_ref_iter = 2;
 lbl->solver_mode = 0;
 lbl->reg_diag = pow(10,-9);
 lbl->symbolic_analysis();
 lbl->numerical_factorization();
 double *x = lbl->solve_only();

 /// Printing results
 // expected x={-2,-2};
 std::cout<<"Solution: ";
// for (int i = 0; i < sizeH; ++i) {
//  std::cout<<x[i]<<",";
// }

std::cout << "GMRES / iterative refinement steps: " << lbl->num_ref_iter << "\n";

// 2) Print raw timing info
if(lbl->psi){
  lbl->psi->print_profiling();
}

lbl->compute_norms();            // compute norms of A, x, b, Ax-b
double be = lbl->backward_error();
std::cout << "Backward error = " << be << std::endl;

if (lbl->L) {
  std::cout << "Number of supernodes: " << lbl->L->nsuper << std::endl;
}

 delete AT;
 delete CT;

 return 0;
}
