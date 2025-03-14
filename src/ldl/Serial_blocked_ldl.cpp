//
// Created by Shujian Qian on 2020-10-29.
//

#include "nasoq/ldl/Serial_blocked_ldl.h"

#include <cassert>

#include <iostream>
#include <vector>
#include <queue>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <chrono>
#include "Eigen/Dense"
#include "Eigen/Cholesky"

#include "nasoq/common/def.h"
#include "nasoq/common/DFS.h"

#include "nasoq/common/Reach.h"

namespace nasoq {
/* ---------------------------------------------------------
   1) BLAS-Like: eigen_dgemm, eigen_dtrsm, eigen_sym_sytrf,
      eigen_dtrsm2, row_reordering, ptranspose
   --------------------------------------------------------- */
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
    Eigen::Map<const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>>
        matA(A, m, k);
    Eigen::Map<const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>>
        matB(B, *ldb, n);
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>>
        matC(C, m, n);
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
    Eigen::Map<const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>>
        Lmat(A, *lda, n);
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>>
        matB(B, *ldb, n);
    Eigen::MatrixXd X_trans = Lmat.triangularView<Eigen::Lower>().solve(matB.transpose());
    Eigen::MatrixXd X      = alp * X_trans.transpose();
    matB = X;
}

static void eigen_sym_sytrf(double* cur, int supWdt, int nSupR,
                            int* nbpivot, double threshold, double* D_block)
{
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>>
        fullBlock(cur, nSupR, supWdt);
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
    Eigen::Map<const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>>
        Tmat(T, supWdt, supWdt);
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>>
        Bmat(B_ptr, nSupR, supWdt);
    Eigen::MatrixXd X_trans = Tmat.triangularView<Eigen::Lower>().solve(Bmat.transpose());
    Eigen::MatrixXd X       = X_trans.transpose();
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

/* ---------------------------------------------------------
   2) Basic (non-pivot) supernodal factor: ldl_left_sn_01
   --------------------------------------------------------- */
bool ldl_left_sn_01(int n, int*c, int*r, double*values,
                           size_t *lC, int*lR, size_t*Li_ptr,double*lValues,
                           double*D,int*blockSet,int supNo,double*timing,
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
        int curCol= (s!=0 ? blockSet[s-1]:0);
        int nxtCol= blockSet[s];
        int supWdt= nxtCol - curCol;
        int nSupR= (int)( Li_ptr[nxtCol] - Li_ptr[curCol] );
        for(int i=(int)Li_ptr[curCol], cnt=0; i<(int)Li_ptr[nxtCol]; i++){
            map[lR[i]]= cnt++;
        }
        for(int i=curCol; i<nxtCol; i++){
            for(int j=c[i]; j<c[i+1]; j++){
                lValues[lC[i] + map[r[j]]] = values[j];
            }
        }
        double *src, *cur=&lValues[lC[curCol]];
#ifndef PRUNE
        top= ereach_sn(supNo,cT,rT,curCol,nxtCol,col2Sup,aTree, xi, xi+supNo);
        assert(top>=0);
        for(int i=top; i< supNo; i++){
            int lSN= xi[i];
#else
        for(int i=0;i< supNo; i++){
            int lSN=i;
#endif
            int cSN= blockSet[lSN];
            int cNSN= blockSet[lSN+1];
            int Li_ptr_cNSN=(int)Li_ptr[cNSN];
            int Li_ptr_cSN =(int)Li_ptr[cSN];
            int nSNRCur= Li_ptr_cNSN - Li_ptr_cSN;
            int supWdts= cNSN - cSN;
            int lb=0,ub=0;
            bool sw=true;
            for(int j=(int)Li_ptr_cSN; j< Li_ptr_cNSN; j++){
                if(lR[j]>=curCol && sw){ lb=j-Li_ptr_cSN; sw=false;}
                if(lR[j]< curCol+ supWdt && !sw){ ub=j-Li_ptr_cSN;}
            }
            int localSupRows= nSNRCur-lb;
            int ndrow1= ub-lb+1;
            int ndrow3= localSupRows- ndrow1;
            src= &lValues[lC[cSN]+lb];
            // multiply by D
            for(int cc=0; cc< supWdts; cc++){
                double tmp=D[cSN+cc];
                for(int rr=0; rr< localSupRows; rr++){
                    trn_diag[ cc* localSupRows + rr ]=
                        tmp * src[ cc*nSNRCur + rr ];
                }
            }
            // gemm
            {
                int M= ndrow3, N= ndrow1, K=supWdts;
                double* srcL= &lValues[lC[cSN] + (ub+1)];
                eigen_dgemm("N","C",&M,&N,&K, one,
                            srcL,&nSNRCur,
                            src ,&nSNRCur,
                            zero,
                            contribs+ ndrow1, &localSupRows);
            }
            // subtract
            for(int i2=0; i2< ndrow1; i2++){
                int colm= map[lR[ Li_ptr_cSN + i2+lb ]];
                for(int j2=i2;j2< localSupRows; j2++){
                    int rowm= map[ lR[ Li_ptr_cSN + j2+lb ]];
                    cur[ colm* localSupRows + rowm ] -=
                        contribs[ i2* localSupRows + j2 ];
                }
            }
        }
        // sytrf
        int nbp2=0;
        eigen_sym_sytrf(cur, supWdt, nSupR, &nbp2, threshold, &D[curCol]);
        // build tri
        int rowNo= nSupR- supWdt;
        for(int l=0;l< supWdt;l++){
            double tmp= cur[l + l*nSupR];
            double *stCol= trn_diag + l*supWdt + l;
            double *cPtr =&cur[l*nSupR + l];
            *stCol= tmp;
            for(int l1=0; l1< supWdt-l-1; l1++){
                *(++stCol)= tmp* *(++cPtr);
            }
        }
        // solve
        eigen_dtrsm2(trn_diag, rowNo, supWdt, &cur[supWdt], nSupR);
        for(int k=0;k< supWdt;k++){
            cur[k*nSupR +k]= 1.0;
        }
    }
    // reorder
    row_reordering(supNo, lC, (int*)Li_ptr, lR, blockSet, nullptr,
                   cT,rT,col2Sup, lValues,
                   std::vector<int>(),nullptr,nullptr,nullptr,nullptr,nullptr);
    delete[]xi; delete[]map;
    delete[]contribs; delete[]trn_diag;
    return true;
}

}