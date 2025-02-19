// yaml_qp_parser.h
#ifndef YAML_QP_PARSER_H
#define YAML_QP_PARSER_H

#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include "qp_problem.h"
#include "matrix_market_parser.h"

static std::string trim_s(const std::string &s){
  size_t b=0; while(b<s.size() && isspace((unsigned char)s[b]))b++;
  size_t e=s.size(); while(e>b && isspace((unsigned char)s[e-1]))e--;
  return s.substr(b,e-b);
}
static std::string detect_key(const std::string &line){
  size_t c=line.find(':');
  if(c==std::string::npos)return "";
  size_t p=line.find('|',c);
  if(p==std::string::npos)return "";
  std::string k=trim_s(line.substr(0,c));
  if(k.size()>1 && k.front()=='\"' && k.back()=='\"'){
    k=k.substr(1,k.size()-2);
  }
  return k;
}
static std::string read_block(std::ifstream &fin, std::string &nk){
  std::ostringstream oss; nk.clear();
  std::string ln;
  while(true){
    std::streampos p=fin.tellg();
    if(!std::getline(fin,ln))break;
    std::string mk=detect_key(ln);
    if(!mk.empty()){ fin.seekg(p); nk=mk; break; }
    oss<<ln<<"\n";
  }
  return oss.str();
}

static bool parse_qp_yaml(const std::string &fn, QPProblem &q){
  std::ifstream fin(fn.c_str());
  if(!fin.good()){
    std::cerr<<"Cannot open "<<fn<<"\n"; 
    return false;
  }
  bool gotQ=false,gotLin=false,gotA=false,gotB=false,gotC=false,gotLb=false,gotUb=false;
  bool Qcoo=false,Lcoo=false,Acoo=false,Bcoo=false,Ccoo=false,Lbcoo=false,Ubcoo=false;
  int Qr=0,Qc=0,Qnz=0;
  int Lr=0,Lc=0,Lnz=0;
  int Ar=0,Ac=0,Anz=0;
  int Br=0,Bc=0,Bnz2=0;
  int Cr=0,Cc=0,Cnz=0;
  int Lbr=0,Lbc=0,Lbnz=0;
  int Ubr=0,Ubc=0,Ubnz=0;
  std::vector<int> Qri,Qci; std::vector<double>Qv;
  std::vector<int> Lri,Lci; std::vector<double>Lv;
  std::vector<int> Ari,Aci; std::vector<double>Av;
  std::vector<int> Bri,Bci; std::vector<double>Bv;
  std::vector<int> Cri,Cci; std::vector<double>Cv;
  std::vector<int> Lbri,Lbci; std::vector<double>Lbv;
  std::vector<int> Ubri,Ubci; std::vector<double>Ubv;

  std::string line,nk;
  while(true){
    if(!std::getline(fin,line))break;
    std::string mk=detect_key(line);
    if(mk.empty())continue;
    std::string blk=read_block(fin,nk);
    if(blk.find("%%MatrixMarket")!=std::string::npos){
      bool coo=false; int rr=0,cc=0,nz=0;
      std::vector<int> rI,cI; std::vector<double> vI;
      bool ok=parse_mm_block(blk,coo,rr,cc,nz,rI,cI,vI);
      if(!ok)return false;
      if(mk=="Quadratic"){
        gotQ=true; Qcoo=coo; Qr=rr; Qc=cc; Qnz=nz;
        Qri=std::move(rI); Qci=std::move(cI); Qv=std::move(vI);
      } else if(mk=="Linear"){
        gotLin=true;Lcoo=coo;Lr=rr;Lc=cc;Lnz=nz;
        Lri=std::move(rI);Lci=std::move(cI);Lv=std::move(vI);
      } else if(mk=="Equality"){
        gotA=true;Acoo=coo;Ar=rr;Ac=cc;Anz=nz;
        Ari=std::move(rI);Aci=std::move(cI);Av=std::move(vI);
      } else if(mk=="Equality bounds"){
        gotB=true;Bcoo=coo;Br=rr;Bc=cc;Bnz2=nz;
        Bri=std::move(rI);Bci=std::move(cI);Bv=std::move(vI);
      } else if(mk=="Inequality"){
        gotC=true;Ccoo=coo;Cr=rr;Cc=cc;Cnz=nz;
        Cri=std::move(rI);Cci=std::move(cI);Cv=std::move(vI);
      } else if(mk=="Inequality l-bounds"){
        gotLb=true;Lbcoo=coo;Lbr=rr;Lbc=cc;Lbnz=nz;
        Lbri=std::move(rI);Lbci=std::move(cI);Lbv=std::move(vI);
      } else if(mk=="Inequality u-bounds"){
        gotUb=true;Ubcoo=coo;Ubr=rr;Ubc=cc;Ubnz=nz;
        Ubri=std::move(rI);Ubci=std::move(cI);Ubv=std::move(vI);
      }
    }
  }
  if(!gotQ){std::cerr<<"No Quadratic!\n";return false;}
  if(Qr!=Qc){std::cerr<<"Quad not square.\n";return false;}
  q.n=Qr;
  q.H=coords_to_csc(Qr,Qc,Qri,Qci,Qv);
  q.q=new double[Qr]; for(int i=0;i<Qr;i++) q.q[i]=0.0;
  if(gotLin){
    if(!Lcoo && Lr==Qr && Lc==1){
      for(int i=0;i< Lr;i++){q.q[i]=Lv[i];}
    } else {
      for(int k=0;k<Lnz;k++){ int r=Lri[k]; if(r>=0 && r<Qr){ q.q[r]=Lv[k]; } }
    }
  }
  if(gotA){
    q.me=Ar;
    q.A= coords_to_csc(Ar,Ac,Ari,Aci,Av);
    if(gotB){
      q.b=new double[Ar]; for(int i=0;i<Ar;i++) q.b[i]=0.0;
      if(!Bcoo && Br==Ar && Bc==1){
        for(int i=0;i<Ar;i++){ q.b[i]=Bv[i]; }
      } else {
        for(int k=0;k<Bnz2;k++){
          int rr=Bri[k]; double val=Bv[k];
          if(rr>=0 && rr<Ar) q.b[rr]=val;
        }
      }
    } else {
      q.b=new double[Ar]; for(int i=0;i<Ar;i++) q.b[i]=0.0;
    }
  } else {
    q.me=0; q.A=new nasoq::CSC; allocateAC(q.A,0,0,-1,0); q.b=nullptr;
  }
  if(gotC){
    q.mi=Cr;
    q.C= coords_to_csc(Cr,Cc,Cri,Cci,Cv);
    q.l=new double[Cr]; q.u=new double[Cr];
    for(int i=0;i<Cr;i++){ q.l[i]=-1e20; q.u[i]=1e20;}
    if(gotLb){
      if(!Lbcoo && Lbr==Cr && Lbc==1){
        for(int i=0;i<Cr;i++){ q.l[i]=Lbv[i]; }
      } else {
        for(int k=0;k<Lbnz;k++){
          int rr=Lbri[k]; double vv=Lbv[k];
          if(rr>=0 && rr<Cr) q.l[rr]=vv;
        }
      }
    }
    if(gotUb){
      if(!Ubcoo && Ubr==Cr && Ubc==1){
        for(int i=0;i<Cr;i++){ q.u[i]=Ubv[i]; }
      } else {
        for(int k=0;k<Ubnz;k++){
          int rr=Ubri[k]; double vv=Ubv[k];
          if(rr>=0 && rr<Cr) q.u[rr]=vv;
        }
      }
    }
  } else {
    q.mi=0; q.C=new nasoq::CSC; allocateAC(q.C,0,0,-1,0); q.l=nullptr; q.u=nullptr;
  }
  return true;
}

#endif

