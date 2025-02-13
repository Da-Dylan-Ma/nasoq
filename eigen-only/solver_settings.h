#ifndef SOLVER_SETTINGS_H
#define SOLVER_SETTINGS_H

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <chrono>
#include "Eigen/Dense"
#include "Eigen/Cholesky"

#include "common/def.h"
#include "common/DFS.h"
#include "common/Reach.h"

namespace nasoq {

/* ---------------------------------------------------------------------------
   1) eigen_dgemm
   2) eigen_dtrsm
   3) eigen_sym_sytrf
   4) eigen_dtrsm2
   5) row_reordering
   6) ptranspose
   7) ldl_left_sn_01
   --------------------------------------------------------------------------- */

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
    Eigen::Map<const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>> matA(A,m,k);
    Eigen::Map<const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>> matB(B,*ldb,n);
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>> matC(C,m,n);
    matC = alp * (matA * matB.transpose()) + bet * matC;
}

static void eigen_dtrsm(const char* /*side*/, const char* /*uplo*/,
                        const char* /*transA*/, const char* /*diag*/,
                        const int *M, const int *N,
                        const double *alpha,
                        const double *A, const int *lda,
                        double *B, const int *ldb)
{
    int m = *M, n = *N;
    double alp = *alpha;
    Eigen::Map<const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>> Lmat(A,*lda,n);
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>> matB(B,*ldb,n);
    Eigen::MatrixXd X_trans = Lmat.triangularView<Eigen::Lower>().solve(matB.transpose());
    Eigen::MatrixXd X = alp * X_trans.transpose();
    matB = X;
}

static void eigen_sym_sytrf(double* cur, int supWdt, int nSupR, int* nbpivot,
                            double threshold, double* D_block)
{
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>> fullBlock(cur,nSupR,supWdt);
    Eigen::MatrixXd A_block = fullBlock.topRows(supWdt);
    Eigen::LDLT<Eigen::MatrixXd,Eigen::Lower> ldlt;
    ldlt.compute(A_block);
    if(ldlt.info() != Eigen::Success){
        *nbpivot = -1;
        return;
    }
    Eigen::MatrixXd Lmat = ldlt.matrixL();
    Eigen::VectorXd Dvec = ldlt.vectorD();
    for(int j=0;j<supWdt;j++){
        for(int i=0;i<supWdt;i++){
            if(i>j) cur[j*nSupR + i] = Lmat(i,j);
            else if(i==j) cur[j*nSupR + i] = 1.0;
            else cur[j*nSupR + i] = 0.0;
        }
        D_block[j] = Dvec(j);
    }
    *nbpivot = 0;
}

static void eigen_dtrsm2(const double* T, int rowNo, int supWdt,
                         double* B_ptr, int nSupR)
{
    (void)rowNo; 
    Eigen::Map<const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>> Tmat(T,supWdt,supWdt);
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>> Bmat(B_ptr,nSupR,supWdt);
    Eigen::MatrixXd X_trans = Tmat.triangularView<Eigen::Lower>().solve(Bmat.transpose());
    Eigen::MatrixXd X = X_trans.transpose();
    Bmat = X;
}

static void row_reordering(int /*supNo*/, size_t* /*lC*/, int* /*Li_ptr*/,
                           int* /*lR*/, int* /*blockSet*/, int* /*atree_sm*/,
                           int* /*cT*/, int* /*rT*/, int* /*col2Sup*/,
                           double* /*lValues*/, std::vector<int> /*perm_req*/,
                           int* /*swap_full*/, int* /*xi*/,
                           int* /*map*/, int* /*ws*/, double* /*contribs*/)
{
    // no-op
}

static CSC* ptranspose(CSC* A,int /*dummy*/,const int* /*perm*/,
                       void* /*workspace*/,int /*dosymm*/,int &status)
{
    if(!A){status=-1; return nullptr;}
    CSC* T=new CSC;
    int m=A->nrow,n=A->ncol;
    T->nrow=n; T->ncol=m; T->stype=-1; T->xtype=1;
    T->nzmax=A->nzmax; T->packed=1; T->sorted=1;
    T->p=new int[T->ncol+1]();
    T->i=new int[T->nzmax];
    T->x=new double[T->nzmax];
    for(int col=0; col<n; col++){
        for(int idx=A->p[col]; idx<A->p[col+1]; idx++){
            int row=A->i[idx];
            if(row>=0 && row<T->ncol){
                T->p[row]++;
            }
        }
    }
    int sum=0;
    for(int i=0;i<T->ncol;i++){
        int tmp=T->p[i];
        T->p[i]=sum;
        sum+=tmp;
    }
    T->p[T->ncol]= A->nzmax;
    std::vector<int> current(T->ncol,0);
    for(int col=0; col<n; col++){
        for(int idx=A->p[col]; idx<A->p[col+1]; idx++){
            int row=A->i[idx];
            double val=A->x[idx];
            int pos=T->p[row] + current[row];
            T->i[pos]= col;
            T->x[pos]= val;
            current[row]++;
        }
    }
    status=0;
    return T;
}

// main factor
static bool ldl_left_sn_01(int n, int*c, int*r, double*values,
                           size_t *lC, int*lR, size_t*Li_ptr,double*lValues,
                           double*D,int*blockSet,int supNo, double*timing,
                           int*aTree,int*cT,int*rT,int*col2Sup,
                           int super_max,int col_max,int &nbpivot,double threshold)
{
    (void)timing;
    nbpivot=0;
    int top=0;
    int* xi=new int[2*supNo]();
    int* map=new int[n]();
    double* contribs=new double[super_max*col_max]();
    double* trn_diag=new double[super_max*col_max]();
    double one[1]={1.0}, zero[1]={0.0};

    for(int s=1; s<= supNo; s++){
        int curCol= (s!=0? blockSet[s-1]:0);
        int nxtCol= blockSet[s];
        int supWdt= nxtCol- curCol;
        int nSupR= (int)(Li_ptr[nxtCol] - Li_ptr[curCol]);
        for(int i=(int)Li_ptr[curCol], cnt=0; i<(int)Li_ptr[nxtCol]; i++){
            map[lR[i]] = cnt++;
        }
        for(int i=curCol;i<nxtCol;i++){
            for(int j=c[i]; j<c[i+1]; j++){
                lValues[lC[i] + map[r[j]]] = values[j];
            }
        }
        double *src,*cur = &lValues[lC[curCol]];
#ifndef PRUNE
        top=ereach_sn(supNo,cT,rT,curCol,nxtCol,col2Sup,aTree, xi, xi+supNo);
        assert(top>=0);
        for(int i=top; i< supNo; i++){
            int lSN= xi[i];
#else
        for(int i=0; i< supNo; i++){
            int lSN= i;
#endif
            int cSN= blockSet[lSN];
            int cNSN= blockSet[lSN+1];
            int Li_ptr_cNSN=(int)Li_ptr[cNSN];
            int Li_ptr_cSN =(int)Li_ptr[cSN];
            int nSNRCur= Li_ptr_cNSN - Li_ptr_cSN;
            int supWdts= cNSN- cSN;
            int lb=0,ub=0;
            bool sw=true;
            for(int j=(int)Li_ptr[cSN]; j<(int)Li_ptr_cNSN; j++){
                if(lR[j]>=curCol && sw){ lb=j-(int)Li_ptr_cSN; sw=false;}
                if(lR[j]<curCol+ supWdt && !sw){ ub=j-(int)Li_ptr_cSN;}
            }
            int localSupRows= nSNRCur - lb;
            int ndrow1= ub- lb +1;
            int ndrow3= localSupRows - ndrow1;
            src= &lValues[ lC[cSN] + lb];
            // multiply by D
            for(int ccc=0; ccc< supWdts; ccc++){
                double tmp= D[cSN + ccc];
                for(int rowx=0; rowx< localSupRows; rowx++){
                    trn_diag[ ccc* localSupRows + rowx ] = tmp* src[ ccc*nSNRCur + rowx];
                }
            }
            {
                int M= ndrow3, N= ndrow1, K= supWdts;
                double* srcL= &lValues[ lC[cSN] + (ub+1) ];
                eigen_dgemm("N","C",&M,&N,&K, one,
                            srcL,&nSNRCur,
                            src ,&nSNRCur,
                            zero,
                            contribs+ ndrow1, &localSupRows);
            }
            for(int i2=0; i2< ndrow1; i2++){
                int colx= map[ lR[ (int)Li_ptr_cSN + i2 + lb] ];
                for(int j2=i2; j2< localSupRows; j2++){
                    int rowx= lR[ (int)Li_ptr_cSN + j2 + lb ];
                    cur[colx*localSupRows + map[rowx]] -= contribs[ i2* localSupRows + j2 ];
                }
            }
        }
        // factor
        eigen_sym_sytrf(cur, supWdt, nSupR, &nbpivot, threshold, &D[curCol]);
        // build tri
        int rowNo= nSupR- supWdt;
        for(int l=0; l< supWdt; l++){
            double tmp= cur[l + l*nSupR];
            double *stCol= trn_diag + l*supWdt + l;
            double *curColPtr= cur + l*nSupR + l;
            *stCol= tmp;
            for(int l1=0;l1< supWdt-l-1; l1++){
                *(++stCol)= tmp* *(++curColPtr);
            }
        }
#ifdef OPENBLAS
#else
        eigen_dtrsm2(trn_diag, rowNo, supWdt, &cur[supWdt], nSupR);
#endif
        for(int k=0;k< supWdt;k++){
            cur[k*nSupR +k] =1.0;
        }
    }
    row_reordering(supNo, lC, (int*)Li_ptr, lR, blockSet, nullptr,
                   cT,rT,col2Sup, lValues, std::vector<int>(),nullptr,nullptr,nullptr,nullptr,nullptr);

    delete[] contribs;
    delete[] trn_diag;
    delete[] xi;
    delete[] map;
    return true;
}

/* ---------------------------------------------------------------------------
   LFactor + profiling_solver_info
   --------------------------------------------------------------------------- */

struct LFactor {
  int nsuper;       
  size_t xsize;     
  int nzmax;        
  size_t *p;        
  int *s;           
  size_t *i_ptr;    
  double *x;        
};

struct profiling_solver_info {
  double fact_time, analysis_time, solve_time, iter_time;
  double ordering_time, update_time, piv_reord;
  double *timing_chol;
  profiling_solver_info(int nt)
   : fact_time(0),analysis_time(0),solve_time(0),iter_time(0),
     ordering_time(0),update_time(0),piv_reord(0)
  {
    timing_chol=new double[4+nt]();
  }
  ~profiling_solver_info(){ delete[] timing_chol; }
  std::chrono::time_point<std::chrono::system_clock> tic(){return std::chrono::system_clock::now();}
  std::chrono::time_point<std::chrono::system_clock> toc(){return std::chrono::system_clock::now();}
  double elapsed_time(std::chrono::time_point<std::chrono::system_clock> beg,
                      std::chrono::time_point<std::chrono::system_clock> end){
    return std::chrono::duration_cast<std::chrono::duration<double>>(end-beg).count();
  }
  void print_profiling(){
    std::cout<<"analysis time: "<<analysis_time<<"; ";
    std::cout<<"fact time: "<<fact_time<<"; ";
    std::cout<<"update time: "<<update_time<<"; ";
    std::cout<<"reordering pivot time: "<<piv_reord<<"; ";
    std::cout<<"solve time: "<<solve_time<<"; ";
  }
};

/* ---------------------------------------------------------------------------
   To match SolverSettings(qp.H, qp.q, qp.A, qp.b, qp.C, qp.l, qp.u)
   --------------------------------------------------------------------------- */
class SolverSettings {
public:
  CSC *H;       // Hessian
  double *q;    // linear
  CSC *A;       // eq
  double *b;    // eq bounds
  CSC *C;       // ineq
  double *l;    // ineq lower
  double *u;    // ineq upper

  CSC *A_ord;  
  CSC *AT_ord; 
  LFactor *L;  
  double *D;   
  int *blockSet;
  int supNo;
  profiling_solver_info *psi;
  double *x;   
  int n;       
  int solver_mode;

  SolverSettings(CSC *H_, double *q_,
                 CSC *A_, double *b_,
                 CSC *C_, double *l_, double *u_)
    : H(H_), q(q_), A(A_), b(b_), C(C_), l(l_), u(u_),
      A_ord(nullptr), AT_ord(nullptr), L(new LFactor),
      D(nullptr), blockSet(nullptr), supNo(1),
      psi(nullptr), x(nullptr), n(0), solver_mode(0)
  {
    default_setting();
    if(A && A->nrow>0) solver_mode=1; // eq
    if(C && C->nrow>0) solver_mode=1; // ineq
    n = H->ncol;
    psi= new profiling_solver_info(1);
  }

  ~SolverSettings(){
    if(A_ord){
      delete[]A_ord->p; delete[]A_ord->i; delete[]A_ord->x; delete A_ord;
    }
    if(AT_ord){
      delete[]AT_ord->p; delete[]AT_ord->i; delete[]AT_ord->x; delete AT_ord;
    }
    if(L){
      if(L->p) delete[]L->p;
      if(L->s) delete[]L->s;
      if(L->i_ptr) delete[]L->i_ptr;
      if(L->x) delete[]L->x;
      delete L;
    }
    if(D) delete[] D;
    if(blockSet) delete[]blockSet;
    if(psi) delete psi;
    if(x) delete[] x;
  }

  void default_setting(){
  }

  // symbolic
  int symbolic_analysis(){
    // copy H -> A_ord
    A_ord= new CSC;
    A_ord->nrow= H->nrow;
    A_ord->ncol= H->ncol;
    A_ord->nzmax= H->nzmax;
    A_ord->p= new int[A_ord->ncol+1];
    A_ord->i= new int[A_ord->nzmax];
    A_ord->x= new double[A_ord->nzmax];
    for(int i=0;i<= (int)H->ncol;i++){
      A_ord->p[i]= H->p[i];
    }
    for(int k=0;k< (int)H->nzmax;k++){
      A_ord->i[k]= H->i[k];
      A_ord->x[k]= H->x[k];
    }
    int st=0;
    AT_ord= ptranspose(A_ord,2,nullptr,nullptr,0, st);

    L->nsuper= supNo;
    L->nzmax= A_ord->nzmax;
    L->p= new size_t[A_ord->ncol+1];
    L->s= new int[A_ord->nzmax];
    L->i_ptr= new size_t[A_ord->ncol+1];
    L->x= new double[A_ord->nzmax];
    for(int i=0;i<= (int)A_ord->ncol;i++){
      L->p[i]= A_ord->p[i];
      L->i_ptr[i]= A_ord->p[i];
    }
    for(int i=0;i< (int)A_ord->nzmax;i++){
      L->s[i]= A_ord->i[i];
      L->x[i]= 0.0;
    }
    D= new double[A_ord->ncol];
    for(int i=0;i<(int)A_ord->ncol;i++){
      D[i]=0.0;
    }
    blockSet= new int[2];
    blockSet[0]= 0;
    blockSet[1]= A_ord->ncol;
    // create x for solution
    x= new double[A_ord->ncol];
    for(int i=0; i< (int)A_ord->ncol; i++){
      x[i]=0.0;
    }
    // TODO: if solver_mode==1, we can also handle eq/ineq
    return 1;
  }

  // factor
  int numerical_factorization(){
    if(!A_ord) return 0;
    int nbpivot=0;
    bool ok= ldl_left_sn_01(A_ord->ncol,
                            A_ord->p, A_ord->i, A_ord->x,
                            L->p, L->s, L->i_ptr, L->x,
                            D, blockSet, supNo,
                            nullptr,nullptr,nullptr,nullptr,nullptr,
                            64,64, nbpivot, 1e-14);
    return (ok?1:0);
  }

  // solve
  double* solve_only(){
    if(!A_ord) return nullptr;
    // TODO: a forward/diag/back approach for the factor
    // forward
    int nvar= (int)A_ord->ncol;
    // use q as the system's RHS; want to keep q separate, copy it to x or so
    for(int i=0;i<nvar;i++){
      x[i]= q[i];
    }
    // forward
    for(int col=0; col<nvar; col++){
      double val= x[col];
      int start= (int)L->i_ptr[col];
      int end= (int)L->i_ptr[col+1];
      double *vals= &L->x[ L->p[col] ];
      for(int p=start+1; p<end; p++){
        int rr= L->s[p];
        x[rr] -= vals[p-start]* val;
      }
    }
    // diag
    for(int i=0; i<nvar; i++){
      if(std::fabs(D[i])<1e-14) D[i]=1e-14;
      x[i]/= D[i];
    }
    // back
    for(int col=nvar-1; col>=0; col--){
      double val= x[col];
      int start=(int)L->i_ptr[col];
      int end=(int)L->i_ptr[col+1];
      double *vals= &L->x[ L->p[col] ];
      for(int p=start+1; p<end; p++){
        int rr= L->s[p];
        x[col] -= vals[p-start]* x[rr];
      }
    }
    return x;
  }

  void reorder_matrix(){
    row_reordering(supNo, L->p, (int*)L->i_ptr, L->s,
                   nullptr,nullptr,nullptr,nullptr,nullptr,
                   L->x,std::vector<int>(),nullptr,nullptr,nullptr,nullptr,nullptr);
  }
};

} // end namespace nasoq

#endif

