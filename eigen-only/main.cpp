#include <iostream>
#include "yaml_qp_parser.h"
#include "solver_settings.h"

using namespace nasoq;

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

  // free
  auto free_csc=[&](CSC* M){
    if(!M)return;
    delete[] M->p; delete[] M->i; delete[] M->x; delete M;
  };
  free_csc(qp.H); free_csc(qp.A); free_csc(qp.C);
  delete[] qp.q; delete[] qp.b; delete[] qp.l; delete[] qp.u;
  return 0;
}

