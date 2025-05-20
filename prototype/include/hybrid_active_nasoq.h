#pragma once

#include "nasoq/nasoq.h"
#include "memory_manager.h"
#include "matrix_free_solvers.h"
#include <vector>
#include <functional>

namespace nasoq {

// Forward declarations
struct CSC;
class MemoryManager;

/**
 * @brief Constraint violation structure for batch processing
 */
struct ConstraintViolation {
    int index;      ///< Constraint index
    double value;   ///< Violation value
    
    // Constructor
    ConstraintViolation(int idx, double val) : index(idx), value(val) {}
    
    // Comparison operator for sorting
    bool operator<(const ConstraintViolation& other) const {
        return value > other.value; // Sort by decreasing violation
    }
};

/**
 * @brief Status codes for solver termination
 */
enum HybridSolverStatus {
    CONVERGED = 0,              ///< Solution converged
    MAX_ITER_REACHED = 1,       ///< Maximum iterations reached
    NUMERICAL_ERROR = 2,        ///< Numerical issues encountered
    INFEASIBLE = 3,             ///< Problem is infeasible
    UNBOUNDED = 4               ///< Problem is unbounded
};

/**
 * @brief Hybrid active-set projection method for QP solving
 * 
 * This class implements a variant of the active-set method that processes
 * constraints in batches and uses matrix-free approaches for KKT solving,
 * making it more suitable for embedded applications.
 */
class HybridActiveNasoq {
private:
    // Problem data
    CSC* H;               ///< Hessian matrix
    CSC* A;               ///< Equality constraint matrix
    CSC* B;               ///< Inequality constraint matrix
    CSC* BT;              ///< Transposed inequality constraint matrix
    double* q;            ///< Linear term
    double* b;            ///< Inequality RHS
    double* a;            ///< Equality RHS
    
    // Problem dimensions
    int H_dim;            ///< Dimension of Hessian matrix
    int A_rows;           ///< Number of equality constraints
    int B_rows;           ///< Number of inequality constraints
    
    // Solution vectors
    double* primal_vars;  ///< Primal variables (size H_dim)
    double* dual_vars_eq; ///< Dual variables for equality constraints (size A_rows)
    double* dual_vars;    ///< Dual variables for inequality constraints (size B_rows)
    
    // Active set management
    std::vector<int> active_set;          ///< Current active set
    std::vector<bool> used_const;         ///< Flags for active constraints
    std::vector<ConstraintViolation> batch_candidates; ///< Candidates for batch update
    int active_set_size;                  ///< Current size of active set
    
    // Solver parameters
    int batch_size;                       ///< Number of constraints to process in each batch
    bool use_matrix_free;                 ///< Whether to use matrix-free KKT solving
    bool use_constraint_relaxation;       ///< Whether to use constraint relaxation
    double relaxation_factor;             ///< Initial relaxation factor
    double relaxation_decay;              ///< Decay rate for relaxation
    double min_relaxation;                ///< Minimum relaxation factor
    double current_relaxation;            ///< Current relaxation factor
    double tolerance;                     ///< Convergence tolerance
    int max_iterations;                   ///< Maximum number of iterations
    
    // Workspace and memory management
    MemoryManager memory_manager;         ///< Memory manager for workspace
    double* workspace;                    ///< Main workspace memory
    std::vector<double> original_b;       ///< Original constraint bounds for relaxation
    bool original_bounds_saved;           ///< Whether original bounds are saved
    int iteration;                        ///< Current iteration number
    
    // Internal solver methods
    
    /**
     * @brief Check for violated constraints and select batch of constraints to add
     * 
     * @param batch The selected batch of constraints will be added to this vector
     * @return int Number of violated constraints
     */
    int find_violated_constraints(std::vector<int>& batch);
    
    /**
     * @brief Calculate optimal step length for a batch of constraints
     * 
     * @param batch Batch of constraint indices
     * @return double Optimal step length
     */
    double calculate_batch_step_length(const std::vector<int>& batch);
    
    /**
     * @brief Update solution with given step length
     * 
     * @param step Step length
     */
    void update_solution(double step);
    
    /**
     * @brief Apply constraint relaxation/tightening
     */
    void apply_constraint_relaxation();
    
    /**
     * @brief Check convergence of current solution
     * 
     * @return bool True if converged
     */
    bool check_convergence();
    
    /**
     * @brief Check feasibility of current solution
     * 
     * @return bool True if feasible
     */
    bool check_feasibility();
    
    /**
     * @brief Create matrix-vector product functions for KKT system
     * 
     * @param kkt_prod Output parameter for KKT matrix-vector product function
     */
    void create_kkt_matrix_vector_product(MatVecProductFn& kkt_prod);
    
    /**
     * @brief Solve KKT system using matrix-free approach
     * 
     * @param rhs Right-hand side vector
     * @param solution Solution vector
     * @return bool Success status
     */
    bool solve_kkt_matrix_free(const double* rhs, double* solution);
    
    /**
     * @brief Solve KKT system using direct approach
     * 
     * @param rhs Right-hand side vector
     * @param solution Solution vector
     * @return bool Success status
     */
    bool solve_kkt_direct(const double* rhs, double* solution);
    
public:
    /**
     * @brief Construct a new Hybrid Active Nasoq solver
     * 
     * @param H_size Dimension of the Hessian matrix
     * @param Hp Column pointers for Hessian (CSC format)
     * @param Hi Row indices for Hessian (CSC format)
     * @param Hx Values for Hessian (CSC format)
     * @param q_in Linear term vector
     * @param A_size1 Number of rows in equality constraint matrix
     * @param A_size2 Number of columns in equality constraint matrix
     * @param Ap Column pointers for equality constraints (CSC format)
     * @param Ai Row indices for equality constraints (CSC format)
     * @param Ax Values for equality constraints (CSC format)
     * @param a_eq Right-hand side for equality constraints
     * @param B_size1 Number of rows in inequality constraint matrix
     * @param B_size2 Number of columns in inequality constraint matrix
     * @param Bp Column pointers for inequality constraints (CSC format)
     * @param Bi Row indices for inequality constraints (CSC format)
     * @param Bx Values for inequality constraints (CSC format)
     * @param b_in Right-hand side for inequality constraints
     */
    HybridActiveNasoq(size_t H_size, int *Hp, int *Hi, double *Hx, double *q_in,
                     size_t A_size1, size_t A_size2, int *Ap, int *Ai, double *Ax, double *a_eq,
                     size_t B_size1, size_t B_size2, int *Bp, int *Bi, double *Bx, double *b_in);
    
    /**
     * @brief Destructor
     */
    ~HybridActiveNasoq();
    
    /**
     * @brief Set solver parameters
     * 
     * @param batch_size Number of constraints to process in each batch
     * @param tolerance Convergence tolerance
     * @param max_iterations Maximum number of iterations
     */
    void set_parameters(int batch_size, double tolerance, int max_iterations);
    
    /**
     * @brief Enable or disable matrix-free KKT solving
     * 
     * @param enable Whether to enable matrix-free solving
     */
    void use_matrix_free_kkt(bool enable);
    
    /**
     * @brief Enable or disable constraint relaxation
     * 
     * @param enable Whether to enable constraint relaxation
     */
    void enable_constraint_relaxation(bool enable);
    
    /**
     * @brief Set relaxation parameters
     * 
     * @param initial_relaxation Initial relaxation factor
     * @param decay_rate Decay rate for relaxation
     * @param min_relaxation Minimum relaxation factor
     */
    void set_relaxation_parameters(double initial_relaxation, double decay_rate, double min_relaxation);
    
    /**
     * @brief Solve the QP problem
     * 
     * @return int Status code from HybridSolverStatus
     */
    int solve();
    
    /**
     * @brief Get the primal solution
     * 
     * @return double* Pointer to primal solution vector
     */
    double* get_primal_solution() { return primal_vars; }
    
    /**
     * @brief Get the dual solution for equality constraints
     * 
     * @return double* Pointer to dual solution vector for equality constraints
     */
    double* get_dual_solution_eq() { return dual_vars_eq; }
    
    /**
     * @brief Get the dual solution for inequality constraints
     * 
     * @return double* Pointer to dual solution vector for inequality constraints
     */
    double* get_dual_solution() { return dual_vars; }
    
    /**
     * @brief Get the number of iterations performed
     * 
     * @return int Number of iterations
     */
    int get_iterations() const { return iteration; }
    
    /**
     * @brief Compute the objective value for the current solution
     * 
     * @return double Objective value
     */
    double compute_objective();
    
    /**
     * @brief Get the active set
     * 
     * @return const std::vector<int>& Reference to active set vector
     */
    const std::vector<int>& get_active_set() const { return active_set; }
};

} // namespace nasoq 