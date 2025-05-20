# NASOQ Hybrid Active-Projection Method Prototype

This prototype implements a new approach to solving quadratic programming (QP) problems, specifically designed for embedded systems with limited resources. It combines the strengths of NASOQ's active-set method with ideas from other optimization approaches.

## Key Features

### 1. Hybrid Active-Projection Method
- Processes constraints in batches rather than one-by-one
- Uses a fixed number of iterations with early termination criteria
- Combines active-set approach with projection methods

### 2. Matrix-Free KKT Solving
- Avoids explicit formation of the KKT matrix
- Uses iterative methods that only require matrix-vector products
- Pre-computes and caches key components

### 3. Constraint Relaxation and Tightening
- Starts with relaxed constraints and gradually tightens them
- Applies different weights to constraints based on importance
- Uses a continuation approach with adjustable parameters

### 4. Memory Optimization
- Custom sparse matrix format optimized for specific problem structures
- Reuse of workspace memory across different operations
- Optional fixed-point arithmetic for systems without efficient floating-point support

## Background

This prototype is inspired by approaches like ReLU-QP (which reformulates the ADMM algorithm as a deep, weight-tied neural network with ReLU-style activation), but adapted specifically for an active-set method like NASOQ, focusing on embedded applications.

## Usage

See the `/examples` directory for demonstration of how to use this prototype.

## Building

```bash
cd nasoq-embedded/nasoq
mkdir build && cd build
cmake -DNASOQ_BUILD_PROTOTYPE=ON ..
make
```

## Status

This is a research prototype and is under active development. It is not recommended for production use yet. 