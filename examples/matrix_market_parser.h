// matrix_market_parser.h
#ifndef MATRIX_MARKET_PARSER_H
#define MATRIX_MARKET_PARSER_H

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include "../include/nasoq/common/def.h"

static bool parse_mm_block(
  const std::string &txt, bool &isCoo, int &nr, int &nc, int &nz,
  std::vector<int> &ri, std::vector<int> &ci, std::vector<double> &val)
{
  isCoo=false; nr=0; nc=0; nz=0;
  std::istringstream iss(txt);
  std::string line;
  while(true) {
    std::streampos p=iss.tellg();
    if(!std::getline(iss,line)) return false;
    if(!line.empty() && line[0]=='%') continue;
    iss.seekg(p);
    break;
  }
  if(!std::getline(iss,line)) return false;
  if(line.find("coordinate")!=std::string::npos) isCoo=true;
  else if(line.find("array")!=std::string::npos) isCoo=false;
  else return false;
  if(!std::getline(iss,line)) return false;
  {
    std::istringstream dss(line);
    if(isCoo) { dss>>nr>>nc>>nz; }
    else { dss>>nr>>nc; nz=nr*nc; }
  }
  ri.resize(nz); ci.resize(nz); val.resize(nz);
  if(isCoo) {
    for(int k=0;k<nz;k++){
      if(!std::getline(iss,line))return false;
      int rr,cc; double vv;
      {std::istringstream ls(line); ls>>rr>>cc>>vv;}
      ri[k]=rr-1; ci[k]=cc-1; val[k]=vv;
    }
  } else {
    for(int i=0;i<nz;i++){
      if(!std::getline(iss,line))return false;
      val[i]=std::stod(line);
      ri[i]= i%nr; // row
      ci[i]= i/nr; // col
    }
  }
  return true;
}

static nasoq::CSC* coords_to_csc(
  int nr,int nc,
  const std::vector<int> &ri,
  const std::vector<int> &ci,
  const std::vector<double> &val)
{
  int nnz=(int)ri.size();
  nasoq::CSC* M=new nasoq::CSC;
  M->nrow=nr; M->ncol=nc; M->nzmax=nnz;
  M->stype=-1; M->xtype=1; M->packed=1; M->sorted=1;
  M->nz=nullptr; M->z=nullptr;
  M->p=new int[nc+1]; 
  M->i=new int[nnz];
  M->x=new double[nnz];
  for(int c=0;c<=nc;c++) M->p[c]=0;
  for(int k=0;k<nnz;k++){ M->p[ci[k]]++; }
  for(int c=0, s=0;c<nc;c++){ int t=M->p[c]; M->p[c]=s; s+=t; }
  M->p[nc]=nnz;
  std::vector<int> nextpos(nc,0);
  for(int c=0;c<nc;c++) nextpos[c]=M->p[c];
  for(int k=0;k<nnz;k++){
    int c=ci[k]; int r=ri[k];
    double v=val[k];
    int dst=nextpos[c]++;
    M->i[dst]=r; M->x[dst]=v;
  }
  return M;
}

#endif

