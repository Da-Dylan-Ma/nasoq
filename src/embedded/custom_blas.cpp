/**
 * @file custom_blas.cpp
 * @brief Implementation of custom BLAS-like functions for NASOQ embedded
 */

#include "nasoq/embedded/custom_blas.h"

namespace nasoq {
namespace embedded {

void dlsolve_blas_nonUnit(int ldm, int ncol, double *M, double *rhs) {
    // Solve L*x = b where L is lower triangular
    // L is stored in column-major format
    
    for (int k = 0; k < ncol; k++) {
        // Check if the diagonal element is non-zero
        if (M[k + k * ldm] != 0.0) {
            // Divide rhs by the diagonal element
            rhs[k] /= M[k + k * ldm];
            
            // Update the rest of the right-hand side
            for (int i = k + 1; i < ncol; i++) {
                rhs[i] -= M[i + k * ldm] * rhs[k];
            }
        }
    }
}

void lSolve_dense_col_sync(int colSize, int col, double *M, double *rhs) {
    // Solve for a single column in the lower triangular system
    // M is stored in column-major format
    
    // Forward substitution for column 'col'
    for (int i = 0; i < colSize; i++) {
        double temp = rhs[i];
        
        // Apply the effect of already computed solution values
        for (int j = 0; j < i; j++) {
            temp -= M[i + j * colSize] * rhs[j];
        }
        
        // Divide by the diagonal element
        rhs[i] = temp / M[i + i * colSize];
    }
}

void dmatvec_blas(int ldm, int nrow, int ncol, double *M, double *vec, double *Mxvec) {
    // Perform matrix-vector multiply: Mxvec += M * vec
    // M is stored in column-major format
    
    // We'll use loop blocking and unrolling for better performance
    
    int col, i, j, l;
    int firstcol = 0;
    
    // Process 8 columns at a time
    while (firstcol < ncol - 7) {
        double *Mcol0 = &M[ldm * firstcol];
        double *Mcol1 = &M[ldm * (firstcol + 1)];
        double *Mcol2 = &M[ldm * (firstcol + 2)];
        double *Mcol3 = &M[ldm * (firstcol + 3)];
        double *Mcol4 = &M[ldm * (firstcol + 4)];
        double *Mcol5 = &M[ldm * (firstcol + 5)];
        double *Mcol6 = &M[ldm * (firstcol + 6)];
        double *Mcol7 = &M[ldm * (firstcol + 7)];
        
        double vi0 = vec[firstcol];
        double vi1 = vec[firstcol + 1];
        double vi2 = vec[firstcol + 2];
        double vi3 = vec[firstcol + 3];
        double vi4 = vec[firstcol + 4];
        double vi5 = vec[firstcol + 5];
        double vi6 = vec[firstcol + 6];
        double vi7 = vec[firstcol + 7];
        
        for (i = 0; i < nrow; i++) {
            Mxvec[i] += vi0 * Mcol0[i] + vi1 * Mcol1[i] + vi2 * Mcol2[i] + vi3 * Mcol3[i] +
                        vi4 * Mcol4[i] + vi5 * Mcol5[i] + vi6 * Mcol6[i] + vi7 * Mcol7[i];
        }
        
        firstcol += 8;
    }
    
    // Process 4 columns at a time
    while (firstcol < ncol - 3) {
        double *Mcol0 = &M[ldm * firstcol];
        double *Mcol1 = &M[ldm * (firstcol + 1)];
        double *Mcol2 = &M[ldm * (firstcol + 2)];
        double *Mcol3 = &M[ldm * (firstcol + 3)];
        
        double vi0 = vec[firstcol];
        double vi1 = vec[firstcol + 1];
        double vi2 = vec[firstcol + 2];
        double vi3 = vec[firstcol + 3];
        
        for (i = 0; i < nrow; i++) {
            Mxvec[i] += vi0 * Mcol0[i] + vi1 * Mcol1[i] + vi2 * Mcol2[i] + vi3 * Mcol3[i];
        }
        
        firstcol += 4;
    }
    
    // Process remaining columns one at a time
    while (firstcol < ncol) {
        double *Mcol = &M[ldm * firstcol];
        double vi = vec[firstcol];
        
        for (i = 0; i < nrow; i++) {
            Mxvec[i] += vi * Mcol[i];
        }
        
        firstcol++;
    }
}

} // namespace embedded
} // namespace nasoq 