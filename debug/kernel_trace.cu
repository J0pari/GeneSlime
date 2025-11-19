#ifndef KERNEL_TRACE_CU
#define KERNEL_TRACE_CU

#include <cuda_runtime.h>
#include <stdio.h>

struct KernelLaunchInfo {
    const char* kernel_name;
    const char* file;
    int line;
    unsigned int grid_x, grid_y, grid_z;
    unsigned int block_x, block_y, block_z;
    size_t shared_mem;
};

__device__ KernelLaunchInfo g_last_launch;

template<typename KernelFunc, typename... Args>
inline void traced_kernel_launch(
    KernelFunc kernel,
    const char* kernel_name,
    const char* file,
    int line,
    dim3 gridDim,
    dim3 blockDim,
    size_t sharedMem,
    cudaStream_t stream,
    Args&&... args
) {
    KernelLaunchInfo info;
    info.kernel_name = kernel_name;
    info.file = file;
    info.line = line;
    info.grid_x = gridDim.x;
    info.grid_y = gridDim.y;
    info.grid_z = gridDim.z;
    info.block_x = blockDim.x;
    info.block_y = blockDim.y;
    info.block_z = blockDim.z;
    info.shared_mem = sharedMem;

    cudaMemcpyToSymbol(g_last_launch, &info, sizeof(KernelLaunchInfo));

    printf("[LAUNCH] %s<<<(%u,%u,%u),(%u,%u,%u),%zu>>> at %s:%d\n",
           kernel_name, gridDim.x, gridDim.y, gridDim.z,
           blockDim.x, blockDim.y, blockDim.z, sharedMem, file, line);

    kernel<<<gridDim, blockDim, sharedMem, stream>>>(std::forward<Args>(args)...);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[FATAL] Kernel launch failed: %s\n", kernel_name);
        printf("        Location: %s:%d\n", file, line);
        printf("        Error: %s (code %d)\n", cudaGetErrorString(err), err);
        printf("        Grid: (%u,%u,%u) Block: (%u,%u,%u) Shared: %zu\n",
               gridDim.x, gridDim.y, gridDim.z,
               blockDim.x, blockDim.y, blockDim.z, sharedMem);
        exit(1);
    }
}

#define LAUNCH_KERNEL(kernel, gridDim, blockDim, sharedMem, streamId, ...) \
    traced_kernel_launch(kernel, #kernel, __FILE__, __LINE__, gridDim, blockDim, sharedMem, streamId, ##__VA_ARGS__)

#define SYNC_CHECK(msg) \
    do { \
        cudaError_t err = cudaDeviceSynchronize(); \
        if (err != cudaSuccess) { \
            KernelLaunchInfo info; \
            cudaMemcpyFromSymbol(&info, g_last_launch, sizeof(KernelLaunchInfo)); \
            printf("[FATAL] Synchronization failed after: %s\n", msg); \
            printf("        Last kernel: %s at %s:%d\n", \
                   info.kernel_name, info.file, info.line); \
            printf("        Grid: (%u,%u,%u) Block: (%u,%u,%u)\n", \
                   info.grid_x, info.grid_y, info.grid_z, \
                   info.block_x, info.block_y, info.block_z); \
            printf("        Error: %s (code %d)\n", cudaGetErrorString(err), err); \
            exit(1); \
        } \
    } while(0)

__global__ void kernel_trace_init() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[TRACE] Kernel tracing system initialized\n");
    }
}

inline void init_kernel_trace() {
    printf("[TRACE] Initializing kernel tracing...\n");
    kernel_trace_init<<<1, 1>>>();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[FATAL] kernel_trace_init launch failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[FATAL] kernel_trace_init sync failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    size_t stack_size = 8192;
    cudaDeviceSetLimit(cudaLimitStackSize, stack_size);

    size_t malloc_heap = 128 * 1024 * 1024;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, malloc_heap);

    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 4);

    cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 32768);

    printf("[TRACE] Stack size: %zu bytes, Heap size: %zu MB\n",
           stack_size, malloc_heap / (1024*1024));
    printf("[TRACE] Dynamic parallelism: sync depth=4, pending launches=32768\n");
}

#endif
