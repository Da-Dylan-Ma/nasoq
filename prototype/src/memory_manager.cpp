#include "memory_manager.h"
#include <cstring>

namespace nasoq {

MemoryManager::MemoryManager(size_t size) : pool_size(size), current_offset(0) {
    main_pool = new double[size];
}

MemoryManager::~MemoryManager() {
    delete[] main_pool;
}

double* MemoryManager::allocate(size_t size, const std::string& tag) {
    // First check if we have an existing allocation of suitable size that's not in use
    for (auto& alloc : allocations) {
        if (!alloc.in_use && alloc.size >= size) {
            alloc.in_use = true;
            alloc.tag = tag;
            return main_pool + alloc.offset;
        }
    }
    
    // If no suitable allocation found, create a new one
    if (current_offset + size <= pool_size) {
        allocations.push_back({current_offset, size, tag, true});
        double* result = main_pool + current_offset;
        current_offset += size;
        return result;
    }
    
    // If we're here, we're out of memory
    throw std::runtime_error("Memory pool exhausted. Requested " + 
                             std::to_string(size) + " doubles, but only " + 
                             std::to_string(pool_size - current_offset) + " available");
}

void MemoryManager::release(double* ptr) {
    size_t offset = ptr - main_pool;
    for (auto& alloc : allocations) {
        if (alloc.offset == offset && alloc.in_use) {
            alloc.in_use = false;
            return;
        }
    }
    throw std::runtime_error("Attempted to release unallocated memory");
}

void MemoryManager::reset() {
    for (auto& alloc : allocations) {
        alloc.in_use = false;
    }
    // No need to actually clear the memory
}

size_t MemoryManager::get_used_memory() const {
    size_t used = 0;
    for (const auto& alloc : allocations) {
        if (alloc.in_use) {
            used += alloc.size;
        }
    }
    return used;
}

// EmbeddedCSC implementation

EmbeddedCSC::EmbeddedCSC(int rows, int cols, int nonzeros, MemoryManager& mem_mgr, bool fixed_point)
    : nrow(rows), ncol(cols), nnz(nonzeros), use_fixed_point(fixed_point), owns_memory(false)
{
    // Allocate memory from the manager
    // Calculate size in doubles (rounded up)
    size_t p_size = (sizeof(int) * (cols + 1) + sizeof(double) - 1) / sizeof(double);
    size_t i_size = (sizeof(int) * nonzeros + sizeof(double) - 1) / sizeof(double);
    
    p = (int*)mem_mgr.allocate(p_size, "CSC_p");
    i = (int*)mem_mgr.allocate(i_size, "CSC_i");
    
    if (fixed_point) {
        size_t x_fixed_size = (sizeof(int32_t) * nonzeros + sizeof(double) - 1) / sizeof(double);
        x_fixed = (int32_t*)mem_mgr.allocate(x_fixed_size, "CSC_x_fixed");
        fixed_point_scale = 1000000; // Default scale, can be adjusted
        x = nullptr;
    } else {
        x = mem_mgr.allocate(nonzeros, "CSC_x");
        x_fixed = nullptr;
    }
}

void EmbeddedCSC::to_fixed_point() {
    if (!use_fixed_point || x_fixed == nullptr || x == nullptr) 
        return;
        
    for (int j = 0; j < nnz; j++) {
        x_fixed[j] = (int32_t)(x[j] * fixed_point_scale);
    }
}

void EmbeddedCSC::to_floating_point() {
    if (!use_fixed_point || x_fixed == nullptr || x == nullptr) 
        return;
        
    for (int j = 0; j < nnz; j++) {
        x[j] = (double)x_fixed[j] / fixed_point_scale;
    }
}

void EmbeddedCSC::multiply(const double* x_vec, double* y_result) const {
    // Initialize result vector to zero
    std::fill(y_result, y_result + nrow, 0.0);
    
    if (use_fixed_point && x_fixed != nullptr) {
        // Fixed-point implementation
        std::vector<int32_t> x_vec_fixed(ncol);
        
        // Convert input vector to fixed-point
        for (int j = 0; j < ncol; j++) {
            x_vec_fixed[j] = (int32_t)(x_vec[j] * fixed_point_scale);
        }
        
        // Perform matrix-vector multiplication in fixed-point
        for (int j = 0; j < ncol; j++) {
            for (int k = p[j]; k < p[j+1]; k++) {
                int64_t prod = (int64_t)x_fixed[k] * x_vec_fixed[j];
                // Convert back to floating-point during accumulation
                y_result[i[k]] += (double)prod / (fixed_point_scale * fixed_point_scale);
            }
        }
    } else {
        // Standard floating-point implementation
        for (int j = 0; j < ncol; j++) {
            for (int k = p[j]; k < p[j+1]; k++) {
                y_result[i[k]] += x[j] * x_vec[j];
            }
        }
    }
}

void EmbeddedCSC::multiply_transposed(const double* x_vec, double* y_result) const {
    // Initialize result vector to zero
    std::fill(y_result, y_result + ncol, 0.0);
    
    if (use_fixed_point && x_fixed != nullptr) {
        // Fixed-point implementation
        std::vector<int32_t> x_vec_fixed(nrow);
        
        // Convert input vector to fixed-point
        for (int i = 0; i < nrow; i++) {
            x_vec_fixed[i] = (int32_t)(x_vec[i] * fixed_point_scale);
        }
        
        // Perform transposed matrix-vector multiplication in fixed-point
        for (int j = 0; j < ncol; j++) {
            for (int k = p[j]; k < p[j+1]; k++) {
                int64_t prod = (int64_t)x_fixed[k] * x_vec_fixed[i[k]];
                // Convert back to floating-point during accumulation
                y_result[j] += (double)prod / (fixed_point_scale * fixed_point_scale);
            }
        }
    } else {
        // Standard floating-point implementation
        for (int j = 0; j < ncol; j++) {
            for (int k = p[j]; k < p[j+1]; k++) {
                y_result[j] += x[k] * x_vec[i[k]];
            }
        }
    }
}

} // namespace nasoq 