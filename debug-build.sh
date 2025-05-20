#!/bin/bash

# Clean build directory if needed
if [ "$1" = "clean" ]; then
    echo "Cleaning build directory..."
    rm -rf build
fi

# Create build directory if it doesn't exist
mkdir -p build

# Change to build directory
cd build

# Configure with embedded implementation and verbose output
cmake -DNASOQ_USE_EMBEDDED=ON -DNASOQ_BUILD_EXAMPLES=ON -DCMAKE_VERBOSE_MAKEFILE=ON ..

# Build the library and examples
make -j$(nproc)

# Check if build was successful
if [ $? -eq 0 ]; then
    echo -e "\nBuild successful!"
    
    # Run NASOQ_Test
    echo -e "\n=== Running NASOQ_Test with embedded implementation ==="
    ./examples/NASOQ_Test
    
    # Also run EMBEDDED_Test for comparison
    echo -e "\n=== Running EMBEDDED_Test (direct test of embedded functions) ==="
    #./examples/EMBEDDED_Test
else
    echo -e "\nBuild failed!"
fi
