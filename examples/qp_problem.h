// qp_problem.h
#ifndef QP_PROBLEM_H
#define QP_PROBLEM_H

#include "../include/nasoq/common/def.h"

struct QPProblem {
  nasoq::CSC *H;  // n x n
  double *q; 
  nasoq::CSC *A; 
  double *b;
  nasoq::CSC *C;
  double *l;
  double *u;
  int n; 
  int me;
  int mi;
  QPProblem() : H(nullptr),q(nullptr),A(nullptr),b(nullptr),C(nullptr),l(nullptr),u(nullptr),
                n(0),me(0),mi(0){}
};

#endif

