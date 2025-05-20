#include "hybrid_active_nasoq.h"
#include <algorithm>
#include <vector>
#include <queue>
#include <cmath>

namespace nasoq {

// Helper function to compute constraint violation for a single constraint
double compute_constraint_violation(const CSC* B, const double* x, const double* b, int constraint_idx) {
    double Bx_val = 0.0;
    int n = B->ncol;
    
    for (int j = 0; j < n; j++) {
        for (int k = B->p[j]; k < B->p[j+1]; k++) {
            if (B->i[k] == constraint_idx) {
                Bx_val += B->x[k] * x[j];
                break;
            }
        }
    }
    
    return Bx_val - b[constraint_idx];
}

// Helper function to identify batch of most violated constraints
std::vector<ConstraintViolation> identify_violated_constraints(
    const CSC* B, const double* x, const double* b, 
    const std::vector<bool>& active_flags, 
    double tolerance) {
    
    std::vector<ConstraintViolation> violations;
    int m = B->nrow;
    
    for (int i = 0; i < m; i++) {
        if (!active_flags[i]) {
            double violation = compute_constraint_violation(B, x, b, i);
            if (violation > tolerance) {
                violations.push_back(ConstraintViolation(i, violation));
            }
        }
    }
    
    // Sort by violation magnitude (decreasing)
    std::sort(violations.begin(), violations.end());
    
    return violations;
}

// Compute the optimal step length for a batch of constraints
double compute_batch_step_length(
    const CSC* H, const CSC* B, 
    const double* x, const double* d, 
    const std::vector<int>& batch, 
    const double* b) {
    
    double alpha_max = std::numeric_limits<double>::max();
    
    for (int idx : batch) {
        // Compute B_i * d
        double Bd = 0.0;
        int n = B->ncol;
        
        for (int j = 0; j < n; j++) {
            for (int k = B->p[j]; k < B->p[j+1]; k++) {
                if (B->i[k] == idx) {
                    Bd += B->x[k] * d[j];
                    break;
                }
            }
        }
        
        if (Bd > 1e-14) {
            // Compute B_i * x - b_i
            double Bx_minus_b = compute_constraint_violation(B, x, b, idx);
            
            // Calculate step length that makes this constraint active
            double alpha = -Bx_minus_b / Bd;
            alpha_max = std::min(alpha_max, alpha);
        }
    }
    
    return alpha_max;
}

// Update the active set with a batch of constraints
void update_active_set_with_batch(
    std::vector<int>& active_set, 
    std::vector<bool>& active_flags, 
    const std::vector<int>& batch) {
    
    for (int idx : batch) {
        if (!active_flags[idx]) {
            active_set.push_back(idx);
            active_flags[idx] = true;
        }
    }
}

// Remove non-binding constraints from the active set
void prune_active_set(
    std::vector<int>& active_set, 
    std::vector<bool>& active_flags, 
    const double* dual_vars, 
    double tolerance) {
    
    std::vector<int> new_active_set;
    
    for (int idx : active_set) {
        if (dual_vars[idx] > -tolerance) {
            new_active_set.push_back(idx);
        } else {
            active_flags[idx] = false;
        }
    }
    
    active_set = new_active_set;
}

} // namespace nasoq 