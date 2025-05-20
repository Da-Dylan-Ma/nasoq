#include "hybrid_active_nasoq.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace nasoq {

// Apply relaxation to inequality constraints
void apply_constraint_relaxation(
    double* b, 
    const std::vector<double>& original_b, 
    double relaxation_factor,
    int m) {
    
    for (int i = 0; i < m; i++) {
        // Apply relaxation - larger values allow more violation
        b[i] = original_b[i] * (1.0 + relaxation_factor);
    }
}

// Apply weighted relaxation - different weights for different constraints
void apply_weighted_relaxation(
    double* b, 
    const std::vector<double>& original_b, 
    const std::vector<double>& weights,
    double relaxation_factor,
    int m) {
    
    for (int i = 0; i < m; i++) {
        // Apply weighted relaxation
        b[i] = original_b[i] * (1.0 + relaxation_factor * weights[i]);
    }
}

// Calculate relaxation weights based on constraint difficulty
std::vector<double> calculate_relaxation_weights(
    const CSC* B, 
    const double* x, 
    const double* b, 
    int m) {
    
    std::vector<double> weights(m);
    
    // Calculate violation for each constraint
    for (int i = 0; i < m; i++) {
        double Bx_val = 0.0;
        int n = B->ncol;
        
        for (int j = 0; j < n; j++) {
            for (int k = B->p[j]; k < B->p[j+1]; k++) {
                if (B->i[k] == i) {
                    Bx_val += B->x[k] * x[j];
                    break;
                }
            }
        }
        
        // Calculate normalized violation
        double violation = std::max(0.0, Bx_val - b[i]);
        double normalized_violation = violation / (std::abs(b[i]) + 1e-10);
        
        // Assign weight based on violation
        weights[i] = std::min(1.0, normalized_violation * 10.0);
    }
    
    return weights;
}

// Adaptive relaxation factor update
double update_relaxation_factor(
    double current_factor, 
    double decay_rate, 
    double min_factor, 
    int current_iter, 
    int max_iter) {
    
    // Simple exponential decay
    double new_factor = current_factor * decay_rate;
    
    // Ensure we don't go below minimum
    return std::max(min_factor, new_factor);
}

// Check if the solution is feasible with original (non-relaxed) constraints
bool check_original_feasibility(
    const CSC* B, 
    const double* x, 
    const std::vector<double>& original_b, 
    double tolerance) {
    
    int m = B->nrow;
    
    for (int i = 0; i < m; i++) {
        double Bx_val = 0.0;
        int n = B->ncol;
        
        for (int j = 0; j < n; j++) {
            for (int k = B->p[j]; k < B->p[j+1]; k++) {
                if (B->i[k] == i) {
                    Bx_val += B->x[k] * x[j];
                    break;
                }
            }
        }
        
        if (Bx_val > original_b[i] + tolerance) {
            return false;
        }
    }
    
    return true;
}

// Compute feasibility gap for current solution
double compute_feasibility_gap(
    const CSC* B, 
    const double* x, 
    const double* b) {
    
    double max_violation = 0.0;
    int m = B->nrow;
    
    for (int i = 0; i < m; i++) {
        double Bx_val = 0.0;
        int n = B->ncol;
        
        for (int j = 0; j < n; j++) {
            for (int k = B->p[j]; k < B->p[j+1]; k++) {
                if (B->i[k] == i) {
                    Bx_val += B->x[k] * x[j];
                    break;
                }
            }
        }
        
        double violation = std::max(0.0, Bx_val - b[i]);
        max_violation = std::max(max_violation, violation);
    }
    
    return max_violation;
}

} // namespace nasoq 