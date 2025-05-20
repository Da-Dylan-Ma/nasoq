#include "matrix_free_solvers.h"
#include <cmath>
#include <algorithm>
#include <utility>
#include <cstring>

namespace nasoq {

// Helper function to compute dot product
double dot_product(const double* a, const double* b, int n) {
    double result = 0.0;
    for (int i = 0; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// Helper function to compute norm
double compute_norm(const double* v, int n) {
    return std::sqrt(dot_product(v, v, n));
}

int conjugate_gradient(const MatVecProductFn& mat_vec_prod,
                      const double* b,
                      double* x,
                      int n,
                      double tol,
                      int max_iter,
                      double* workspace) {
    // Workspace allocation:
    // workspace[0:n-1] = residual r
    // workspace[n:2n-1] = search direction p
    // workspace[2n:3n-1] = matrix-vector product Ap
    double* r = workspace;
    double* p = workspace + n;
    double* Ap = workspace + 2*n;
    
    // Initialize residual r = b - Ax
    std::copy(b, b + n, r);
    mat_vec_prod(x, Ap);  // Compute Ax into Ap temporarily
    for (int i = 0; i < n; i++) {
        r[i] -= Ap[i];
    }
    
    // Initialize p = r
    std::copy(r, r + n, p);
    
    double rsold = dot_product(r, r, n);
    double initial_res = std::sqrt(rsold);
    
    if (initial_res < tol) {
        return 0;  // Already converged
    }
    
    for (int iter = 0; iter < max_iter; iter++) {
        // Compute Ap = A*p
        mat_vec_prod(p, Ap);
        
        // Compute step size alpha = rsold / (p' * Ap)
        double pAp = dot_product(p, Ap, n);
        if (std::abs(pAp) < 1e-14) {
            // Matrix is singular or nearly singular
            return -1;
        }
        
        double alpha = rsold / pAp;
        
        // Update solution x = x + alpha*p
        for (int i = 0; i < n; i++) {
            x[i] += alpha * p[i];
        }
        
        // Update residual r = r - alpha*Ap
        for (int i = 0; i < n; i++) {
            r[i] -= alpha * Ap[i];
        }
        
        // Compute new dot product
        double rsnew = dot_product(r, r, n);
        
        // Check convergence
        if (std::sqrt(rsnew) < tol * initial_res) {
            return iter + 1;
        }
        
        // Update search direction p = r + beta*p
        double beta = rsnew / rsold;
        for (int i = 0; i < n; i++) {
            p[i] = r[i] + beta * p[i];
        }
        
        rsold = rsnew;
    }
    
    return -1;  // Did not converge within max_iter
}

int preconditioned_conjugate_gradient(const MatVecProductFn& mat_vec_prod,
                                     const PreconditionerFn& precond,
                                     const double* b,
                                     double* x,
                                     int n,
                                     double tol,
                                     int max_iter,
                                     double* workspace) {
    // Workspace allocation:
    // workspace[0:n-1] = residual r
    // workspace[n:2n-1] = search direction p
    // workspace[2n:3n-1] = matrix-vector product Ap
    // workspace[3n:4n-1] = preconditioned residual z
    double* r = workspace;
    double* p = workspace + n;
    double* Ap = workspace + 2*n;
    double* z = workspace + 3*n;
    
    // Initialize residual r = b - Ax
    std::copy(b, b + n, r);
    mat_vec_prod(x, Ap);  // Compute Ax into Ap temporarily
    for (int i = 0; i < n; i++) {
        r[i] -= Ap[i];
    }
    
    // Apply preconditioner z = M^-1 * r
    precond(r, z);
    
    // Initialize p = z
    std::copy(z, z + n, p);
    
    double rz = dot_product(r, z, n);
    double initial_res = std::sqrt(dot_product(r, r, n));
    
    if (initial_res < tol) {
        return 0;  // Already converged
    }
    
    for (int iter = 0; iter < max_iter; iter++) {
        // Compute Ap = A*p
        mat_vec_prod(p, Ap);
        
        // Compute step size alpha = rz / (p' * Ap)
        double pAp = dot_product(p, Ap, n);
        if (std::abs(pAp) < 1e-14) {
            // Matrix is singular or nearly singular
            return -1;
        }
        
        double alpha = rz / pAp;
        
        // Update solution x = x + alpha*p
        for (int i = 0; i < n; i++) {
            x[i] += alpha * p[i];
        }
        
        // Update residual r = r - alpha*Ap
        for (int i = 0; i < n; i++) {
            r[i] -= alpha * Ap[i];
        }
        
        // Check convergence
        double res_norm = std::sqrt(dot_product(r, r, n));
        if (res_norm < tol * initial_res) {
            return iter + 1;
        }
        
        // Apply preconditioner z = M^-1 * r
        precond(r, z);
        
        double rz_new = dot_product(r, z, n);
        
        // Update search direction p = z + beta*p
        double beta = rz_new / rz;
        for (int i = 0; i < n; i++) {
            p[i] = z[i] + beta * p[i];
        }
        
        rz = rz_new;
    }
    
    return -1;  // Did not converge within max_iter
}

// Simplified MINRES for saddle point systems
int minres_saddle_point(const MatVecProductFn& H_prod,
                        const MatVecProductFn& A_prod,
                        const MatVecProductFn& AT_prod,
                        const double* f,
                        const double* g,
                        double* x,
                        double* y,
                        int n1,
                        int n2,
                        double tol,
                        int max_iter,
                        double* workspace) {
    // Total problem size
    int n = n1 + n2;
    
    // Allocate workspace for the combined system
    double* r = workspace;               // Residual (size n)
    double* v = workspace + n;           // Orthogonal direction (size n)
    double* v_prev = workspace + 2*n;    // Previous orthogonal direction (size n)
    double* Kv = workspace + 3*n;        // Matrix-vector product (size n)
    double* combined_sol = workspace + 4*n; // Combined solution vector [x; y] (size n)
    double* combined_rhs = workspace + 5*n; // Combined right-hand side [f; g] (size n)
    
    // Create a combined system operator
    auto K_prod = [&](const double* v, double* Kv) {
        const double* v1 = v;        // First n1 components
        const double* v2 = v + n1;   // Last n2 components
        double* Kv1 = Kv;            // First n1 components of result
        double* Kv2 = Kv + n1;       // Last n2 components of result
        
        // Compute upper block: Kv1 = H*v1 + A'*v2
        H_prod(v1, Kv1);
        AT_prod(v2, workspace + 6*n); // Temporary storage
        for (int i = 0; i < n1; i++) {
            Kv1[i] += workspace[6*n + i];
        }
        
        // Compute lower block: Kv2 = A*v1
        A_prod(v1, Kv2);
    };
    
    // Initialize the combined solution and right-hand side
    for (int i = 0; i < n1; i++) {
        combined_sol[i] = x[i];
        combined_rhs[i] = f[i];
    }
    for (int i = 0; i < n2; i++) {
        combined_sol[n1 + i] = y[i];
        combined_rhs[n1 + i] = g[i];
    }
    
    // Initialize residual r = b - Ax
    std::copy(combined_rhs, combined_rhs + n, r);
    K_prod(combined_sol, Kv);  // Compute K*sol into Kv temporarily
    for (int i = 0; i < n; i++) {
        r[i] -= Kv[i];
    }
    
    // Initialize v = r / ||r||
    double beta = compute_norm(r, n);
    if (beta < tol) {
        // Already converged
        std::copy(combined_sol, combined_sol + n1, x);
        std::copy(combined_sol + n1, combined_sol + n, y);
        return 0;
    }
    
    double inv_beta = 1.0 / beta;
    for (int i = 0; i < n; i++) {
        v[i] = r[i] * inv_beta;
    }
    
    std::fill(v_prev, v_prev + n, 0.0);
    
    double gamma_prev = 0.0;
    double gamma = 0.0;
    double delta = 0.0;
    double epsilon = 0.0;
    
    // MINRES iteration
    for (int iter = 0; iter < max_iter; iter++) {
        // Compute Kv = K*v
        K_prod(v, Kv);
        
        // Update coefficients for the recurrence
        double alpha = dot_product(v, Kv, n);
        
        // Orthogonalize against the previous two vectors (simplified 3-term recurrence)
        for (int i = 0; i < n; i++) {
            Kv[i] -= alpha * v[i];
            if (iter > 0) {
                Kv[i] -= beta * v_prev[i];
            }
        }
        
        double beta_new = compute_norm(Kv, n);
        
        // Update solution
        double gamma_old = gamma;
        gamma = 1.0 / std::sqrt(alpha*alpha + beta*beta);
        delta = gamma * alpha;
        epsilon = gamma * beta;
        
        for (int i = 0; i < n; i++) {
            combined_sol[i] += delta * beta * v[i];
        }
        
        // Prepare for next iteration
        std::copy(v, v + n, v_prev);
        if (beta_new < 1e-14) {
            // Breakdown (lucky convergence)
            break;
        }
        
        inv_beta = 1.0 / beta_new;
        for (int i = 0; i < n; i++) {
            v[i] = Kv[i] * inv_beta;
        }
        
        beta = beta_new;
        
        // Check convergence (approximated residual)
        if (beta * std::abs(gamma) < tol) {
            // Copy back solution
            std::copy(combined_sol, combined_sol + n1, x);
            std::copy(combined_sol + n1, combined_sol + n, y);
            return iter + 1;
        }
    }
    
    // Did not converge within max_iter, but copy back the current solution
    std::copy(combined_sol, combined_sol + n1, x);
    std::copy(combined_sol + n1, combined_sol + n, y);
    
    return -1;
}

PreconditionerFn create_diagonal_preconditioner(const double* diag, int n) {
    // Allocate memory for the reciprocal of the diagonal
    double* diag_inv = new double[n];
    
    for (int i = 0; i < n; i++) {
        // Handle zeros with a small regularization
        if (std::abs(diag[i]) < 1e-14) {
            diag_inv[i] = 1.0;
        } else {
            diag_inv[i] = 1.0 / diag[i];
        }
    }
    
    // Return a function that applies the diagonal preconditioner
    // and captures the allocated memory
    return [diag_inv, n](const double* r, double* z) {
        for (int i = 0; i < n; i++) {
            z[i] = r[i] * diag_inv[i];
        }
    };
}

std::pair<PreconditionerFn, std::unique_ptr<double[]>> 
create_block_diagonal_preconditioner(const double* H_diag, 
                                    int n1,
                                    const MatVecProductFn& A_prod,
                                    const MatVecProductFn& AT_prod,
                                    int n2) {
    // Allocate memory for the preconditioner data
    // We need:
    // 1. Inverse of H diagonal (n1)
    // 2. Approximate Schur complement diagonal (n2)
    // 3. Workspace for matrix-vector products (n1 + n2)
    int total_size = n1 + n2 + n1 + n2;
    auto precond_data = std::make_unique<double[]>(total_size);
    
    double* H_diag_inv = precond_data.get();
    double* S_diag_inv = H_diag_inv + n1;
    double* workspace = S_diag_inv + n2;
    
    // Compute inverse of H diagonal
    for (int i = 0; i < n1; i++) {
        // Regularize zeros
        if (std::abs(H_diag[i]) < 1e-14) {
            H_diag_inv[i] = 1.0;
        } else {
            H_diag_inv[i] = 1.0 / H_diag[i];
        }
    }
    
    // Approximate Schur complement diagonal S = A*H_diag_inv*A'
    // We compute this column by column
    std::fill(S_diag_inv, S_diag_inv + n2, 0.0);
    
    for (int j = 0; j < n1; j++) {
        // Create a unit vector
        std::fill(workspace, workspace + n1, 0.0);
        workspace[j] = H_diag_inv[j];
        
        // Compute A*ej*H_diag_inv[j]
        A_prod(workspace, workspace + n1);
        
        // Accumulate into S_diag_inv
        for (int i = 0; i < n2; i++) {
            S_diag_inv[i] += workspace[n1 + i] * workspace[n1 + i];
        }
    }
    
    // Invert S_diag
    for (int i = 0; i < n2; i++) {
        // Regularize zeros
        if (std::abs(S_diag_inv[i]) < 1e-14) {
            S_diag_inv[i] = 1.0;
        } else {
            S_diag_inv[i] = 1.0 / S_diag_inv[i];
        }
    }
    
    // Create the preconditioner function
    auto precond_fn = [n1, n2, H_diag_inv=H_diag_inv, S_diag_inv=S_diag_inv, 
                     workspace=workspace, A_prod, AT_prod]
                    (const double* r, double* z) {
        const double* r1 = r;          // First n1 components of residual
        const double* r2 = r + n1;     // Last n2 components of residual
        double* z1 = z;                // First n1 components of result
        double* z2 = z + n1;           // Last n2 components of result
        double* temp = workspace;      // Temporary workspace
        
        // First solve S*z2 = r2
        for (int i = 0; i < n2; i++) {
            z2[i] = r2[i] * S_diag_inv[i];
        }
        
        // Then compute z1 = H_diag_inv * (r1 - A'*z2)
        AT_prod(z2, temp);
        for (int i = 0; i < n1; i++) {
            z1[i] = (r1[i] - temp[i]) * H_diag_inv[i];
        }
    };
    
    return std::make_pair(precond_fn, std::move(precond_data));
}

} // namespace nasoq 