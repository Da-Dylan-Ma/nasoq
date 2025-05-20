#pragma once

#include <functional>
#include <vector>
#include <memory>

namespace nasoq {

/**
 * @brief Function type for matrix-vector product callbacks
 * 
 * This function type represents a callback that computes y = A*x
 * without explicitly forming the matrix A.
 */
using MatVecProductFn = std::function<void(const double* x, double* y)>;

/**
 * @brief Solve a linear system using the Conjugate Gradient method.
 * 
 * This function solves the linear system Ax = b using the Conjugate
 * Gradient method, which only requires matrix-vector products via
 * the provided callback function.
 * 
 * @param mat_vec_prod Function to compute matrix-vector product
 * @param b Right-hand side vector
 * @param x Initial guess and solution vector
 * @param n Problem dimension
 * @param tol Convergence tolerance
 * @param max_iter Maximum number of iterations
 * @param workspace Pre-allocated workspace of size at least 3*n
 * @return int Number of iterations performed or -1 if not converged
 */
int conjugate_gradient(const MatVecProductFn& mat_vec_prod,
                       const double* b,
                       double* x,
                       int n,
                       double tol,
                       int max_iter,
                       double* workspace);

/**
 * @brief Function type for preconditioner callbacks
 * 
 * This function type represents a callback that applies a preconditioner 
 * M^-1 to a vector z = M^-1 * r.
 */
using PreconditionerFn = std::function<void(const double* r, double* z)>;

/**
 * @brief Solve a linear system using the Preconditioned Conjugate Gradient method.
 * 
 * This function solves the linear system Ax = b using the Preconditioned Conjugate
 * Gradient method, which uses a preconditioner to accelerate convergence.
 * 
 * @param mat_vec_prod Function to compute matrix-vector product
 * @param precond Function to apply the preconditioner
 * @param b Right-hand side vector
 * @param x Initial guess and solution vector
 * @param n Problem dimension
 * @param tol Convergence tolerance
 * @param max_iter Maximum number of iterations
 * @param workspace Pre-allocated workspace of size at least 4*n
 * @return int Number of iterations performed or -1 if not converged
 */
int preconditioned_conjugate_gradient(const MatVecProductFn& mat_vec_prod,
                                     const PreconditionerFn& precond,
                                     const double* b,
                                     double* x,
                                     int n,
                                     double tol,
                                     int max_iter,
                                     double* workspace);

/**
 * @brief Solve a saddle-point system using the MINRES method.
 * 
 * This function solves saddle-point systems of the form:
 * [ H  A' ] [ x ] = [ f ]
 * [ A  0  ] [ y ]   [ g ]
 * using the MINRES method, which is suitable for indefinite systems.
 * 
 * @param H_prod Function to compute H*x product
 * @param A_prod Function to compute A*x product
 * @param AT_prod Function to compute A'*y product
 * @param f Upper part of right-hand side
 * @param g Lower part of right-hand side
 * @param x Upper part of solution vector
 * @param y Lower part of solution vector
 * @param n1 Dimension of x
 * @param n2 Dimension of y
 * @param tol Convergence tolerance
 * @param max_iter Maximum number of iterations
 * @param workspace Pre-allocated workspace
 * @return int Number of iterations performed or -1 if not converged
 */
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
                        double* workspace);

/**
 * @brief Create a diagonal preconditioner from the diagonal of a matrix.
 * 
 * @param diag Array containing the diagonal elements of the matrix
 * @param n Matrix dimension
 * @return PreconditionerFn Function that applies the diagonal preconditioner
 */
PreconditionerFn create_diagonal_preconditioner(const double* diag, int n);

/**
 * @brief Create a block-diagonal preconditioner for saddle-point systems.
 * 
 * This function creates a block-diagonal preconditioner for systems of the form:
 * [ H  A' ] [ x ] = [ f ]
 * [ A  0  ] [ y ]   [ g ]
 * using the Schur complement approximation.
 * 
 * @param H_diag Diagonal of the H matrix
 * @param n1 Dimension of H
 * @param A_prod Function to compute A*x product
 * @param AT_prod Function to compute A'*y product
 * @param n2 Dimension of the lower block
 * @return std::pair<MatVecProductFn, double*> Preconditioner function and allocated memory
 */
std::pair<PreconditionerFn, std::unique_ptr<double[]>> 
create_block_diagonal_preconditioner(const double* H_diag, 
                                    int n1,
                                    const MatVecProductFn& A_prod,
                                    const MatVecProductFn& AT_prod,
                                    int n2);

} // namespace nasoq 