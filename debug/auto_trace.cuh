#ifndef AUTO_TRACE_CUH
#define AUTO_TRACE_CUH

#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 1
#endif

#if DEBUG_LEVEL >= 1
    #define DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
    #define DEBUG_PRINT(...)
#endif

#if DEBUG_LEVEL >= 2
    #define DEBUG_VERBOSE(...) printf(__VA_ARGS__)
#else
    #define DEBUG_VERBOSE(...)
#endif

#if DEBUG_LEVEL >= 3
    #define DEBUG_TRACE(...) printf("[TRACE] " __VA_ARGS__)
#else
    #define DEBUG_TRACE(...)
#endif

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("[CUDA ERROR] %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define CHECK_LAST() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            printf("[CUDA ERROR] %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define SYNC_AND_CHECK() \
    do { \
        CHECK_CUDA(cudaDeviceSynchronize()); \
    } while(0)

__device__ void debug_assert(bool condition, const char* msg, const char* file, int line) {
    if (!condition) {
        printf("[ASSERT FAILED] %s at %s:%d\n", msg, file, line);
    }
}

#define DEVICE_ASSERT(cond, msg) debug_assert(cond, msg, __FILE__, __LINE__)

#endif
