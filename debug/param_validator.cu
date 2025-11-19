#ifndef PARAM_VALIDATOR_CU
#define PARAM_VALIDATOR_CU

#include <cuda_runtime.h>
#include <stdio.h>

// Validate any pointer
inline bool validate_pointer(const char* name, void* ptr, bool must_be_device, const char* file, int line) {
    if (ptr == nullptr) {
        printf("[PARAM_NULL] %s is NULL at %s:%d\n", name, file, line);
        return false;
    }

    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
        printf("[PARAM_ERROR] %s (%p) query failed: %s at %s:%d\n",
               name, ptr, cudaGetErrorString(err), file, line);
        cudaGetLastError();
        return false;
    }

    const char* type_str;
    bool is_valid = true;

    switch(attr.type) {
        case cudaMemoryTypeUnregistered: type_str = "unregistered"; is_valid = !must_be_device; break;
        case cudaMemoryTypeHost: type_str = "host"; is_valid = !must_be_device; break;
        case cudaMemoryTypeDevice: type_str = "device"; break;
        case cudaMemoryTypeManaged: type_str = "managed"; break;
        default: type_str = "unknown"; is_valid = false; break;
    }

    if (!is_valid) {
        printf("[PARAM_WRONG_TYPE] %s (%p) is %s (expected device) at %s:%d\n",
               name, ptr, type_str, file, line);
        return false;
    }

    printf("[PARAM_OK] %s=%p (%s) at %s:%d\n", name, ptr, type_str, file, line);
    return true;
}

// Print struct size
template<typename T>
inline void print_struct_layout(const char* name) {
    printf("[STRUCT] %s: size=%zu bytes, align=%zu\n", name, sizeof(T), alignof(T));
}

// Validate integer range
inline bool validate_int_range(const char* name, int value, int min, int max, const char* file, int line) {
    if (value < min || value > max) {
        printf("[PARAM_RANGE] %s=%d out of range [%d,%d] at %s:%d\n",
               name, value, min, max, file, line);
        return false;
    }
    printf("[PARAM_OK] %s=%d at %s:%d\n", name, value, file, line);
    return true;
}

// Validate launch config
inline bool validate_launch_config(dim3 grid, dim3 block, size_t shared_mem, const char* kernel_name, const char* file, int line) {
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    bool valid = true;

    if (block.x * block.y * block.z > (unsigned)prop.maxThreadsPerBlock) {
        printf("[LAUNCH_ERROR] %s: block size %ux%ux%u = %u exceeds max %d at %s:%d\n",
               kernel_name, block.x, block.y, block.z, block.x*block.y*block.z,
               prop.maxThreadsPerBlock, file, line);
        valid = false;
    }

    if (grid.x > (unsigned)prop.maxGridSize[0] ||
        grid.y > (unsigned)prop.maxGridSize[1] ||
        grid.z > (unsigned)prop.maxGridSize[2]) {
        printf("[LAUNCH_ERROR] %s: grid size %ux%ux%u exceeds max [%d,%d,%d] at %s:%d\n",
               kernel_name, grid.x, grid.y, grid.z,
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2], file, line);
        valid = false;
    }

    if (shared_mem > prop.sharedMemPerBlock) {
        printf("[LAUNCH_ERROR] %s: shared memory %zu exceeds max %zu at %s:%d\n",
               kernel_name, shared_mem, prop.sharedMemPerBlock, file, line);
        valid = false;
    }

    if (valid) {
        printf("[LAUNCH_OK] %s: grid=(%u,%u,%u) block=(%u,%u,%u) shared=%zu at %s:%d\n",
               kernel_name, grid.x, grid.y, grid.z, block.x, block.y, block.z, shared_mem, file, line);
    }

    return valid;
}

#define VALIDATE_DEVICE_PTR(ptr) \
    validate_pointer(#ptr, (void*)(ptr), true, __FILE__, __LINE__)

#define VALIDATE_HOST_PTR(ptr) \
    validate_pointer(#ptr, (void*)(ptr), false, __FILE__, __LINE__)

#define VALIDATE_INT_RANGE(val, min, max) \
    validate_int_range(#val, val, min, max, __FILE__, __LINE__)

#define VALIDATE_LAUNCH_CONFIG(grid, block, shared, kernel) \
    validate_launch_config(grid, block, shared, #kernel, __FILE__, __LINE__)

#define PRINT_STRUCT(type) \
    print_struct_layout<type>(#type)

#define BEGIN_KERNEL_VALIDATION(kernel_name) \
    do { \
        printf("\n=== VALIDATING: %s ===\n", kernel_name); \
        printf("Location: %s:%d\n", __FILE__, __LINE__); \
        cudaError_t _pending_err = cudaGetLastError(); \
        if (_pending_err != cudaSuccess) { \
            printf("[WARN] Pending CUDA error: %s\n", cudaGetErrorString(_pending_err)); \
        } \
    } while(0)

#define END_KERNEL_VALIDATION() \
    do { \
        printf("======================\n\n"); \
    } while(0)

#endif
