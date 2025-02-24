rm -rf build
cmake -DNASOQ_BLAS_BACKEND=OpenBLAS -DNASOQ_USE_CLAPACK=ON -DCMAKE_BUILD_TYPE=Release  -S . -B build
cmake --build build --config Release -j 6
