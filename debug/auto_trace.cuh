#ifndef AUTO_TRACE_CUH
#define AUTO_TRACE_CUH

// Auto-tracing kernel launch system
// Include this FIRST in any .cu file to automatically trace all kernel launches

#include "kernel_trace.cu"

// Preprocessor hack: redefine kernel launch syntax
// We intercept kernel<<<grid, block>>>(args...) by preprocessing it

// Helper to convert standard kernel<<<grid,block,shared,stream>>>(args) to traced version
// This works by creating wrapper functions that the preprocessor can rewrite

#ifdef __CUDACC__
// In CUDA mode, define automatic tracing wrappers

// Note: We can't directly intercept the <<<>>> operator at preprocessor level
// Instead, we require using a simple macro for all launches:
// Instead of: kernel<<<grid, block>>>(args...);
// Use:        LAUNCH(kernel, grid, block, 0, 0, args...);
// Or simplified: LAUNCH_SIMPLE(kernel, grid, block, args...);

// These macros provide full tracing automatically:
#define LAUNCH(kernel, grid, block, shared, stream, ...) \
    traced_kernel_launch(kernel, #kernel, __FILE__, __LINE__, grid, block, shared, stream, ##__VA_ARGS__)

#define LAUNCH_SIMPLE(kernel, grid, block, ...) \
    traced_kernel_launch(kernel, #kernel, __FILE__, __LINE__, grid, block, 0, 0, ##__VA_ARGS__)

#endif // __CUDACC__

#endif // AUTO_TRACE_CUH
