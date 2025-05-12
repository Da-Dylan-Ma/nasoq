
#include <cmath>
#include <iostream>
#include <nasoq/nasoq.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <fstream>

#include "yaml_qp_parser.h"
#include "../smp-format/io.h"

static void print_qp_debug(const QPProblem &qp) {
    // Hessian H
    if (qp.H) {
        std::cout << "H: " << qp.H->nrow << " x " << qp.H->ncol
                  << " nnz=" << qp.H->nzmax << "\n";
        int show_cols = std::min((int) qp.H->ncol, 5);
        for (int col = 0; col < show_cols; col++) {
            int start = qp.H->p[col];
            int end = qp.H->p[col + 1];
            for (int idx = start; idx < end; idx++) {
                int row = qp.H->i[idx];
                if (row < 5) {
                    double val = qp.H->x[idx];
                    std::cout << "  H(" << row << "," << col << ")=" << val << "\n";
                }
            }
        }
    }

    // linear q
    if (qp.q) {
        std::cout << "q: [";
        int show_q = std::min(qp.n, 5);
        for (int i = 0; i < show_q; i++) {
            std::cout << qp.q[i] << " ";
        }
        std::cout << ((qp.n > 5) ? "...]\n" : "]\n");
    }

    // equality constraints A, b
    if (qp.A) {
        std::cout << "A: " << qp.A->nrow << " x " << qp.A->ncol
                  << " nnz=" << qp.A->nzmax << "\n";
        int show_cols = std::min((int) qp.A->ncol, 5);
        for (int col = 0; col < show_cols; col++) {
            int start = qp.A->p[col];
            int end = qp.A->p[col + 1];
            for (int idx = start; idx < end; idx++) {
                int row = qp.A->i[idx];
                if (row < 5) {
                    double val = qp.A->x[idx];
                    std::cout << "  A(" << row << "," << col << ")=" << val << "\n";
                }
            }
        }
    }
    if (qp.b) {
        std::cout << "b: [";
        int show_b = std::min(qp.me, 5);
        for (int i = 0; i < show_b; i++) {
            std::cout << qp.b[i] << " ";
        }
        std::cout << ((qp.me > 5) ? "...]\n" : "]\n");
    }

    // inequality constraints C, l, u
    if (qp.C) {
        std::cout << "C: " << qp.C->nrow << " x " << qp.C->ncol
                  << " nnz=" << qp.C->nzmax << "\n";
        int show_cols = std::min((int) qp.C->ncol, 5);
        for (int col = 0; col < show_cols; col++) {
            int start = qp.C->p[col];
            int end = qp.C->p[col + 1];
            for (int idx = start; idx < end; idx++) {
                int row = qp.C->i[idx];
                if (row < 5) {
                    double val = qp.C->x[idx];
                    std::cout << "  C(" << row << "," << col << ")=" << val << "\n";
                }
            }
        }
    }
    if (qp.l && qp.u) {
        int show_ineq = std::min(qp.mi, 5);
        std::cout << "l: [";
        for (int i = 0; i < show_ineq; i++) {
            std::cout << qp.l[i] << " ";
        }
        std::cout << ((qp.mi > 5) ? "...]\n" : "]\n");
        std::cout << "u: [";
        for (int i = 0; i < show_ineq; i++) {
            std::cout << qp.u[i] << " ";
        }
        std::cout << ((qp.mi > 5) ? "...]\n" : "]\n");
    }
}


nasoq::CSC *build_stacked_constraint(const nasoq::CSC *C,
                              const double *l,
                              const double *u,
                              int mi,      // #rows in C
                              int n,       // #cols in C
                              double *&stacked_b) // returned array of size 2*mi
{
    if(!C || !l || !u || mi <= 0 || n <= 0){
        std::cerr << "Invalid input to build_stacked_constraint\n";
        return nullptr;
    }
    // Construct the new matrix stackedC of dimension (2*mi) x n
    nasoq::CSC *stackedC = new nasoq::CSC;
    stackedC->nrow = 2 * mi;
    stackedC->ncol = n;
    // We assume the nonzero pattern is simply doubled:
    stackedC->nzmax = 2 * (C->p[n]); // p[n] = # of nonzeros in original C
    stackedC->p = new int[n+1];
    stackedC->i = new int[stackedC->nzmax];
    stackedC->x = new double[stackedC->nzmax];

    // Also set metadata
    stackedC->stype = 0;
    stackedC->xtype = 1;
    stackedC->sorted = 1;
    stackedC->packed = 1;

    // We'll fill stackedC->p[j] = #nnz so far up to col j
    stackedC->p[0] = 0;
    int nnz_so_far = 0;

    // Step 1: copy C col j
    // Step 2: copy -C col j (row indices shifted by mi)
    for(int j = 0; j < n; j++){
        // number of nonzeros in col j of original C
        int col_start = C->p[j];
        int col_end   = C->p[j+1];

        // copy C->(col j)
        for(int pC = col_start; pC < col_end; pC++){
            int rowC   = C->i[pC];    // row index in [0..mi-1]
            double valC = C->x[pC];
            // same row index for the top block
            stackedC->i[nnz_so_far] = rowC;
            stackedC->x[nnz_so_far] = valC;
            nnz_so_far++;
        }
        // copy -C->(col j), shifting row indices by mi
        for(int pC = col_start; pC < col_end; pC++){
            int rowC   = C->i[pC];
            double valC = C->x[pC];
            // row index is rowC + mi
            stackedC->i[nnz_so_far] = rowC + mi;
            stackedC->x[nnz_so_far] = -valC; // negative
            nnz_so_far++;
        }
        stackedC->p[j+1] = nnz_so_far;
    }

    // Build the stacked_b array: dimension 2*mi
    stacked_b = new double[2 * mi];
    // top half = u, bottom half = -l
    for(int i = 0; i < mi; i++){
        stacked_b[i]       = u[i];   //  C*x <= u
        stacked_b[i + mi]  = -l[i];  // -C*x <= -l
    }

    return stackedC;
}


/*
 * Minimizing 1/2 x^THx + q^Tx + C; Cx <= d
 * H and C are sparse nasoq::CSC matrices
 * q and d are dense arrays
 */

int main(int argc, char *argv[]){
    if (argc < 2) {
        std::cout << "usage: " << argv[0] << " <qp_smp.yml>\n";
        return 1;
    }
    std::string fname = argv[1];
    QPProblem qp;
    bool ok = parse_qp_yaml(fname, qp);
    if (!ok) {
        std::cerr << "parse failed\n";
        return 2;
    }
    std::cout << "parsed n=" << qp.n << ", me=" << qp.me << ", mi=" << qp.mi << "\n";

    print_qp_debug(qp);

    if (!qp.H || !qp.q) {
        std::cerr << "Error: Hessian or q is null\n";
        return false;
    }
    // If me>0, we expect qp.A, qp.b not null
    // If mi>0, we expect qp.C, and at least one bound pointer (e.g. qp.u)

    // 2) Prepare arguments for the Nasoq constructor.
    //    We assume 'C x <= u' is your ineq form, and 'A x = b' is eq.

    // The Hessian part
    size_t H_size  = qp.n;
    int   *Hp      = qp.H->p;
    int   *Hi      = qp.H->i;
    double *Hx     = qp.H->x;
    double *q_in   = qp.q;

    // The equality constraints
    size_t A_size1 = qp.me;  // #rows = me
    size_t A_size2 = qp.n;   // #cols = n
    int   *Ap      = (qp.A) ? qp.A->p : nullptr;
    int   *Ai      = (qp.A) ? qp.A->i : nullptr;
    double *Ax     = (qp.A) ? qp.A->x : nullptr;
    double *a_eq   = qp.b;   // pointer to eq RHS (length me)

    // Default: single-sided approach if only "u" is given
    size_t B_size1 = qp.mi;
    size_t B_size2 = qp.n;
    int   *Bp      = (qp.C) ? qp.C->p : nullptr;
    int   *Bi      = (qp.C) ? qp.C->i : nullptr;
    double *Bx     = (qp.C) ? qp.C->x : nullptr;
    double *b_ineq = qp.u;

    nasoq::CSC *stC = nullptr;
    double *stb = nullptr;

    // If we have both l and u (two-sided ineq), we build a stacked matrix
    if(qp.l && qp.u && qp.mi > 0 && qp.C){
        stC = build_stacked_constraint(qp.C, qp.l, qp.u, qp.mi, qp.n, stb);
        if(!stC || !stb){
            std::cerr << "Error creating stacked ineq constraints\n";
            return 4;
        }
        B_size1 = 2 * qp.mi;
        B_size2 = qp.n;
        Bp = stC->p;
        Bi = stC->i;
        Bx = stC->x;
        b_ineq = stb;
    }

    // 3) Construct Nasoq
    nasoq::Nasoq *qm = new nasoq::Nasoq(
            H_size, Hp, Hi, Hx, q_in,
            A_size1, A_size2, Ap, Ai, Ax, a_eq,
            B_size1, B_size2, Bp, Bi, Bx, b_ineq
    );
 qm->diag_perturb=pow(10,-9);
 qm->eps_abs=pow(10,-3);
 qm->max_iter = 100;
 qm->variant = nasoq::PREDET;
 int converged = qm->solve();

 /// Printing results
 if(converged)
  std::cout<<"The problem is converged:" << converged << std::endl;

 std::cout << "cons_sat_norm: " << qm->cons_sat_norm << std::endl;
 std::cout << "lag_res: " << qm->lag_res << std::endl;
 std::cout << "non_negativity_infn: " << qm->non_negativity_infn << std::endl;
 std::cout << "complementarity_infn: " << qm->complementarity_infn << std::endl;
 std::cout << "eps_abs: " << qm->eps_abs << std::endl;

 // expected x={0.4,1.2};
 auto *x = qm->primal_vars;
 std::cout<<"Primal variables: ";
 for (int i = 0; i < H_size; ++i) {
  std::cout<<x[i]<<",";
 }

 std::cout << std::endl;

 // expected z = {1.6,0,0,0}
 std::cout<<"\nDual variables: ";
 auto *z = qm->dual_vars;
 for (int i = 0; i < A_size1; ++i) {
  std::cout<<z[i]<<",";
 }

 std::cout << std::endl;

    // === CSV File Handling ===
    std::ofstream csv_file;
    std::string csv_filename = "results.csv";
    bool file_exists = std::ifstream(csv_filename).good(); // Check if file exists

    csv_file.open(csv_filename, std::ios::app); // Open in append mode

    // Write header if the file is newly created
    if (!file_exists) {
        csv_file << "Test Case,eps_abs,Convergence,cons_sat_norm,lag_res,non_negativity_infn,complementarity_infn,Iterations\n";
    }

    // Append results
    csv_file << fname << ","
             << qm->eps_abs << ","
             << converged << ","
             << qm->cons_sat_norm << ","
             << qm->lag_res << ","
             << qm->non_negativity_infn << ","
             << qm->complementarity_infn << ","
             << qm->num_iter << "\n";

    csv_file.close();


    delete qm;
    delete [] Hp;
    delete [] Hi;
    delete [] Hx;

    // Clean up stacked or not
    if(stC){
        delete [] stC->p;
        delete [] stC->i;
        delete [] stC->x;
        delete stC;
    }
    if(stb){
        delete [] stb;
    }
    return 0;
}

