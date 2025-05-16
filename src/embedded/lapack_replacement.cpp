/**
 * @file lapack_replacement.cpp
 * @brief Implementation of LAPACK replacements for bare-metal ARM Cortex
 */

#include "nasoq/embedded/lapack_replacement.h"
#include <cmath>
#include <algorithm>

namespace nasoq {
namespace embedded {

void dsytrf(const char* uplo, const int* n, double* a, const int* lda,
            int* ipiv, double* work, const int* lwork, int* info) {
    // Check for invalid inputs
    if (uplo[0] != 'U' && uplo[0] != 'u' && uplo[0] != 'L' && uplo[0] != 'l') {
        *info = -1;
        return;
    }
    if (*n < 0) {
        *info = -2;
        return;
    }
    if (*lda < std::max(1, *n)) {
        *info = -4;
        return;
    }
    if (*lwork < std::max(1, *n) && *lwork != -1) {
        *info = -7;
        return;
    }

    // Successful parameter check
    *info = 0;

    // Quick return if possible
    if (*n == 0) {
        work[0] = 1.0;
        return;
    }

    // Determine workspace size
    if (*lwork == -1) {
        work[0] = static_cast<double>(*n);
        return;
    }

    bool upper = (uplo[0] == 'U' || uplo[0] == 'u');

    // Initialize ipiv
    for (int i = 0; i < *n; i++) {
        ipiv[i] = i;
    }

    // Simple Bunch-Kaufman algorithm for symmetric indefinite factorization
    // This is a basic implementation that can be optimized further
    
    const double alpha = (1.0 + std::sqrt(17.0)) / 8.0;
    
    if (upper) {
        // Upper triangular case
        // TODO: Implement upper triangular case
    } else {
        // Lower triangular case
        for (int k = 0; k < *n; k++) {
            // Find pivot
            int kp = k;
            double absakk = std::abs(a[k + k * (*lda)]);
            
            // Look for a larger pivot
            double colmax = 0.0;
            int imax = k;
            for (int i = k + 1; i < *n; i++) {
                double curval = std::abs(a[i + k * (*lda)]);
                if (curval > colmax) {
                    colmax = curval;
                    imax = i;
                }
            }
            
            if (colmax * alpha >= absakk) {
                kp = imax;
                absakk = std::abs(a[imax + imax * (*lda)]);
            }
            
            // Apply interchange to columns k and kp
            if (kp != k) {
                // Swap diagonal elements
                std::swap(a[k + k * (*lda)], a[kp + kp * (*lda)]);
                
                // Swap off-diagonal elements in column k
                for (int i = k + 1; i < kp; i++) {
                    std::swap(a[i + k * (*lda)], a[kp + i * (*lda)]);
                }
                
                // Swap off-diagonal elements in column kp
                for (int i = kp + 1; i < *n; i++) {
                    std::swap(a[i + k * (*lda)], a[i + kp * (*lda)]);
                }
                
                // Update ipiv
                std::swap(ipiv[k], ipiv[kp]);
            }
            
            // Check for singularity
            if (a[k + k * (*lda)] == 0.0) {
                *info = k + 1;
                return;
            }
            
            // Update the trailing submatrix
            if (k < *n - 1) {
                // Compute elements below the diagonal
                double rkk = 1.0 / a[k + k * (*lda)];
                for (int i = k + 1; i < *n; i++) {
                    a[i + k * (*lda)] *= rkk;
                }
                
                // Rank-1 update of trailing submatrix
                for (int i = k + 1; i < *n; i++) {
                    for (int j = k + 1; j <= i; j++) {
                        a[i + j * (*lda)] -= a[i + k * (*lda)] * a[j + k * (*lda)] * a[k + k * (*lda)];
                    }
                }
            }
        }
    }
}

void lapmt(const bool* forwrd, const int* m, const int* n,
           double* a, const int* lda, const int* perm) {
    // Use static memory for efficiency
    static double* temp = nullptr;
    static int temp_size = 0;
    static int* permCopy = nullptr;
    static int perm_size = 0;
    
    // Only reallocate if needed
    if (temp == nullptr || temp_size < *m) {
        delete[] temp;
        temp_size = *m;
        temp = new double[temp_size];
    }
    
    if (permCopy == nullptr || perm_size < *n) {
        delete[] permCopy;
        perm_size = *n;
        permCopy = new int[perm_size];
    }
    
    // Make a copy of perm that we can modify
    for (int i = 0; i < *n; i++) {
        permCopy[i] = perm[i];
    }
    
    if (*forwrd) {
        // Forward permutation
        for (int j = 0; j < *n; j++) {
            if (permCopy[j] != j) {
                int k = j;
                // Save column j
                for (int i = 0; i < *m; i++) {
                    temp[i] = a[i + j * (*lda)];
                }
                
                // Follow the permutation cycle
                while (permCopy[k] != j) {
                    int nextk = permCopy[k];
                    
                    // Copy column permCopy[k] to column k
                    for (int i = 0; i < *m; i++) {
                        a[i + k * (*lda)] = a[i + nextk * (*lda)];
                    }
                    
                    // Mark as done
                    permCopy[k] = k;
                    k = nextk;
                }
                
                // Copy saved column j to column k
                for (int i = 0; i < *m; i++) {
                    a[i + k * (*lda)] = temp[i];
                }
                
                // Mark as done
                permCopy[k] = k;
            }
        }
    } else {
        // Backward permutation
        for (int j = *n - 1; j >= 0; j--) {
            if (permCopy[j] != j) {
                int k = permCopy[j];
                
                // Save column k
                for (int i = 0; i < *m; i++) {
                    temp[i] = a[i + k * (*lda)];
                }
                
                // Follow the permutation cycle in reverse
                while (k != j) {
                    int nextk = permCopy[k];
                    
                    // Copy column permCopy[nextk] to column k
                    for (int i = 0; i < *m; i++) {
                        a[i + k * (*lda)] = a[i + nextk * (*lda)];
                    }
                    
                    // Mark as done
                    permCopy[k] = k;
                    k = nextk;
                }
                
                // Copy saved column to column j
                for (int i = 0; i < *m; i++) {
                    a[i + j * (*lda)] = temp[i];
                }
                
                // Mark as done
                permCopy[j] = j;
            }
        }
    }
    
    // Don't delete temp or permCopy - we'll reuse them
}

} // namespace embedded
} // namespace nasoq 