#include <iostream>
#include "yaml_qp_parser.h"
#include "solver_settings.h"

using namespace nasoq;


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


int main(int argc,char**argv){
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

  SolverSettings sset(qp.H,qp.q, qp.A, qp.b, qp.C, qp.l, qp.u);
  sset.symbolic_analysis();
  int fac_ok= sset.numerical_factorization();
  if(!fac_ok){
    std::cerr<<"factor failed\n";
    return 3;
  }
  double *sol= sset.solve_only();
  std::cout<<"solution[0..4]: ";
  for(int i=0;i< (qp.n>5?5:qp.n); i++){
    std::cout<<sol[i]<<" ";
  }
  std::cout<<"\n";

  auto free_csc=[&](CSC* M){
    if(!M)return;
    delete[] M->p; delete[] M->i; delete[] M->x; delete M;
  };
  free_csc(qp.H); free_csc(qp.A); free_csc(qp.C);
  delete[] qp.q; delete[] qp.b; delete[] qp.l; delete[] qp.u;
  return 0;
}

