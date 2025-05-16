# NASOQ Embedded Implementation

This extension provides bare-metal ARM Cortex implementation of NASOQ by replacing external BLAS/LAPACK dependencies with optimized C++ implementations.

## Building the Embedded Version

To build NASOQ with the embedded implementation:

```bash
git clone https://github.com/sympiler/nasoq.git
cd nasoq
cmake -DNASOQ_BLAS_BACKEND=OpenBLAS -DCMAKE_BUILD_TYPE=Release -DNASOQ_EMBEDDED=ON -S . -B build
cmake --build build --config Release -j 6
```

This will build NASOQ with the embedded BLAS/LAPACK replacements without requiring any external BLAS or LAPACK libraries.

> **Note:** 
> 1. Even when using embedded mode, you should still specify a BLAS backend (`OpenBLAS` recommended) to satisfy CMake dependencies, although the actual BLAS implementation won't be used at runtime.
> 2. Do NOT specify `-DNASOQ_USE_CLAPACK=ON` when in embedded mode, as it will cause CMake errors.

## What's Included

The embedded implementation provides:

1. Replacements for core BLAS functions:
   - `dgemm` (matrix-matrix multiplication)
   - `dtrsm` (triangular solver with multiple right-hand sides)
   - `dgemv` (matrix-vector multiplication)
   - `dscal` (vector scaling)
   - `dsyr` (symmetric rank-1 update)

2. Replacements for LAPACK functions:
   - `dsytrf` (symmetric factorization)
   - `lapmt` (matrix permutation)

3. Custom BLAS-like functions:
   - `dlsolve_blas_nonUnit` (non-unit triangular solver)
   - `lSolve_dense_col_sync` (row-synchronized triangular solve)
   - `dmatvec_blas` (matrix-vector multiplication)

## Optimization for ARM Cortex

These implementations can be further optimized for specific ARM Cortex processors by:

1. Using NEON SIMD instructions for vectorized operations
2. Implementing cache-friendly blocking for matrix operations
3. Using loop unrolling (already implemented in some functions)
4. Taking advantage of ARM-specific optimizations

## Integration

The embedded implementation is conditionally included using the `NASOQ_EMBEDDED` flag. When this flag is defined:

1. The BLAS/LAPACK functions are redirected to our embedded implementations
2. No external BLAS/LAPACK libraries are required
3. The code is more suitable for embedded platforms without standard libraries 

## Setting Up in a New Project

To set up the embedded version in a new project:

1. Clone the NASOQ repository:
   ```bash
   git clone https://github.com/sympiler/nasoq.git
   ```

2. Copy the embedded implementation files to your project:
   - `include/nasoq/embedded/*.h` - Header files 
   - `src/embedded/*.cpp` - Implementation files

3. When using CMake:
   ```cmake
   # Define the NASOQ_EMBEDDED flag to enable embedded mode
   add_definitions(-DNASOQ_EMBEDDED)
   
   # Include the NASOQ headers
   include_directories(${NASOQ_INCLUDE_DIR})
   
   # Link with your target
   target_link_libraries(your_target nasoq)
   ```

4. For a bare-metal ARM Cortex build, you'll need to configure your toolchain appropriately:
   ```bash
   cmake -DNASOQ_EMBEDDED=ON -DCMAKE_TOOLCHAIN_FILE=your_arm_toolchain.cmake -S . -B build
   ``` 