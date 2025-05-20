#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <algorithm>
#include <memory>

namespace nasoq {

/**
 * @brief Memory manager for efficient memory allocation and reuse.
 * 
 * This class provides a memory pool that can be reused across different parts
 * of the solver, reducing memory allocation overhead and fragmentation.
 */
class MemoryManager {
private:
    // Memory pool
    double* main_pool;
    size_t pool_size;
    
    // Current allocation pointer
    size_t current_offset;
    
    // Allocation records for reuse
    struct Allocation {
        size_t offset;
        size_t size;
        std::string tag; // For debugging/tracking
        bool in_use;
    };
    
    std::vector<Allocation> allocations;
    
public:
    /**
     * @brief Construct a new Memory Manager
     * 
     * @param size Size of the memory pool in number of doubles
     */
    MemoryManager(size_t size);
    
    /**
     * @brief Destroy the Memory Manager and free all allocated memory
     */
    ~MemoryManager();
    
    /**
     * @brief Allocate memory from the pool
     * 
     * @param size Number of doubles to allocate
     * @param tag Identifier for the allocation (for debugging)
     * @return double* Pointer to the allocated memory
     * @throws std::runtime_error if the memory pool is exhausted
     */
    double* allocate(size_t size, const std::string& tag);
    
    /**
     * @brief Release memory back to the pool for reuse
     * 
     * @param ptr Pointer to the memory to release
     * @throws std::runtime_error if the pointer was not allocated from this pool
     */
    void release(double* ptr);
    
    /**
     * @brief Reset the entire pool, marking all allocations as available
     */
    void reset();
    
    /**
     * @brief Get the total size of the memory pool
     * 
     * @return size_t Size in number of doubles
     */
    size_t get_pool_size() const { return pool_size; }
    
    /**
     * @brief Get the currently used memory in the pool
     * 
     * @return size_t Used memory in number of doubles
     */
    size_t get_used_memory() const;
};

/**
 * @brief Specialized sparse matrix format for embedded systems.
 * 
 * This struct provides a more memory-efficient representation of sparse matrices
 * with optional fixed-point arithmetic support for systems without efficient
 * floating-point operations.
 */
struct EmbeddedCSC {
    int nrow;    ///< Number of rows
    int ncol;    ///< Number of columns
    int nnz;     ///< Number of non-zero elements
    
    // Traditional CSC components
    int* p;     ///< Column pointers (size: ncol+1)
    int* i;     ///< Row indices (size: nnz)
    double* x;  ///< Values (size: nnz)
    
    // Optional fixed-point representation
    bool use_fixed_point;              ///< Whether to use fixed-point arithmetic
    int32_t* x_fixed;                  ///< Fixed-point values (size: nnz)
    int fixed_point_scale;             ///< Scale factor for fixed-point conversion
    
    // Memory is owned elsewhere (typically by MemoryManager)
    bool owns_memory;                  ///< Whether this struct owns its memory
    
    /**
     * @brief Construct a new EmbeddedCSC matrix using memory from a MemoryManager
     * 
     * @param rows Number of rows
     * @param cols Number of columns
     * @param nonzeros Number of non-zero elements
     * @param mem_mgr Reference to the memory manager
     * @param fixed_point Whether to use fixed-point arithmetic
     */
    EmbeddedCSC(int rows, int cols, int nonzeros, MemoryManager& mem_mgr, bool fixed_point = false);
    
    /**
     * @brief Convert from floating-point to fixed-point representation
     */
    void to_fixed_point();
    
    /**
     * @brief Convert from fixed-point to floating-point representation
     */
    void to_floating_point();
    
    /**
     * @brief Perform matrix-vector multiplication y = A*x
     * 
     * @param x Input vector
     * @param y Output vector (must be pre-allocated)
     */
    void multiply(const double* x, double* y) const;
    
    /**
     * @brief Perform transposed matrix-vector multiplication y = A'*x
     * 
     * @param x Input vector
     * @param y Output vector (must be pre-allocated)
     */
    void multiply_transposed(const double* x, double* y) const;
};

} // namespace nasoq 