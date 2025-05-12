#include <iostream>
#include <fstream>
#include <cmath>
#include <nasoq/nasoq.h>
#include "../codegen/qp_data_static.h"

using namespace nasoq;
using namespace qp_data;

CSC *build_stacked_constraint(const CSC *C, const double *l, const double *u, int mi, int n, double *&stacked_b) {
    CSC *stackedC = new CSC;
    stackedC->nrow = 2 * mi;
    stackedC->ncol = n;
    stackedC->nzmax = 2 * C->p[n];
    stackedC->p = new int[n + 1];
    stackedC->i = new int[stackedC->nzmax];
    stackedC->x = new double[stackedC->nzmax];
    stackedC->stype = 0;
    stackedC->xtype = 1;
    stackedC->sorted = 1;
    stackedC->packed = 1;

    stackedC->p[0] = 0;
    int nnz = 0;
    for (int j = 0; j < n; j++) {
        int start = C->p[j], end = C->p[j + 1];
        for (int k = start; k < end; k++) {
            stackedC->i[nnz] = C->i[k];
            stackedC->x[nnz] = C->x[k];
            nnz++;
        }
        for (int k = start; k < end; k++) {
            stackedC->i[nnz] = C->i[k] + mi;
            stackedC->x[nnz] = -C->x[k];
            nnz++;
        }
        stackedC->p[j + 1] = nnz;
    }

    stacked_b = new double[2 * mi];
    for (int i = 0; i < mi; i++) {
        stacked_b[i] = u[i];
        stacked_b[i + mi] = -l[i];
    }
    return stackedC;
}

int main() {
    size_t n = sizeof(H_p) / sizeof(H_p[0]) - 1;
    size_t me = sizeof(b) / sizeof(b[0]);
    size_t mi = sizeof(u) / sizeof(u[0]);

    CSC H, A, C;
    H.nrow = H.ncol = n;
    H.nzmax = sizeof(H_x) / sizeof(H_x[0]);
    H.p = const_cast<int *>(H_p);
    H.i = const_cast<int *>(H_i);
    H.x = const_cast<double *>(H_x);
    H.stype = 0;
    H.xtype = 1;
    H.sorted = 1;
    H.packed = 1;

    A.nrow = me;
    A.ncol = n;
    A.nzmax = sizeof(A_x) / sizeof(A_x[0]);
    A.p = const_cast<int *>(A_p);
    A.i = const_cast<int *>(A_i);
    A.x = const_cast<double *>(A_x);
    A.stype = 0;
    A.xtype = 1;
    A.sorted = 1;
    A.packed = 1;

    C.nrow = mi;
    C.ncol = n;
    C.nzmax = sizeof(C_x) / sizeof(C_x[0]);
    C.p = const_cast<int *>(C_p);
    C.i = const_cast<int *>(C_i);
    C.x = const_cast<double *>(C_x);
    C.stype = 0;
    C.xtype = 1;
    C.sorted = 1;
    C.packed = 1;

    double *q_in = const_cast<double *>(q);
    double *a_eq = const_cast<double *>(b);
    double *b_ineq = const_cast<double *>(u);

    CSC *B_mat = &C;
    double *b_used = b_ineq;

    CSC *stackedC = nullptr;
    double *stacked_b = nullptr;
    if (l && u) {
        stackedC = build_stacked_constraint(&C, l, u, mi, n, stacked_b);
        B_mat = stackedC;
        b_used = stacked_b;
    }

    Nasoq *solver = new Nasoq(
            n, H.p, H.i, H.x, q_in,
            me, n, A.p, A.i, A.x, a_eq,
            B_mat->nrow, B_mat->ncol, B_mat->p, B_mat->i, B_mat->x, b_used
    );

    solver->diag_perturb = 1e-9;
    solver->eps_abs = 1e-3;
    solver->max_iter = 100;
    solver->variant = PREDET;

    int status = solver->solve();

    std::cout << "Converged: " << status << "\n";
    std::cout << "cons_sat_norm: " << solver->cons_sat_norm << "\n";
    std::cout << "lag_res: " << solver->lag_res << "\n";
    std::cout << "non_negativity_infn: " << solver->non_negativity_infn << "\n";
    std::cout << "complementarity_infn: " << solver->complementarity_infn << "\n";

    std::cout << "Primal: ";
    for (int i = 0; i < n; i++) std::cout << solver->primal_vars[i] << " ";
    std::cout << "\n";

    std::ofstream csv("results.csv", std::ios::app);
    csv << "static_qp,"
        << solver->eps_abs << ","
        << status << ","
        << solver->cons_sat_norm << ","
        << solver->lag_res << ","
        << solver->non_negativity_infn << ","
        << solver->complementarity_infn << ","
        << solver->num_iter << "\n";
    csv.close();

    delete solver;
    if (stackedC) {
        delete[] stackedC->p;
        delete[] stackedC->i;
        delete[] stackedC->x;
        delete stackedC;
    }
    delete[] stacked_b;

    return 0;
}
