#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "../core/types.cuh"
#include "../utils/tile_ops.cuh"

using namespace nvcuda;

// ============================================================================
// WMMA Tensor Core Operations (16×16×16 FP16)
// ============================================================================

#if __CUDA_ARCH__ >= 700

__device__ void wmma_matmul_16x16x16(
    const half* A,      // [16 × 16]
    const half* B,      // [16 × 16]
    float* C,           // [16 × 16] accumulator
    int lda,
    int ldb,
    int ldc
) {
    // Declare fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize accumulator to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Load matrices into fragments
    wmma::load_matrix_sync(a_frag, A, lda);
    wmma::load_matrix_sync(b_frag, B, ldb);

    // Perform matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store result
    wmma::store_matrix_sync(C, c_frag, ldc, wmma::mem_row_major);
}

__global__ void wmma_perception_kernel(
    const float* ca_concentration,  // [GRID_CELLS_2D × CHANNELS]
    const half* perception_weights, // [NUM_HEADS × CHANNELS × HIDDEN_DIM]
    float* perception_output,       // [NUM_HEADS × GRID_CELLS_2D × HIDDEN_DIM]
    int grid_cells
) {
    // Each block processes one grid cell for all heads
    int cell_idx = blockIdx.x;
    if (cell_idx >= grid_cells) return;

    // Each warp handles one head
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (warp_id >= NUM_HEADS) return;

    // Load concentration for this cell (CHANNELS values)
    __shared__ half input_tile[NUM_HEADS][WMMA_M][WMMA_K];
    __shared__ float output_tile[NUM_HEADS][WMMA_M][WMMA_N];

    // Convert FP32 concentration to FP16 for WMMA
    if (lane_id < CHANNELS) {
        float val = ca_concentration[cell_idx * CHANNELS + lane_id];
        input_tile[warp_id][0][lane_id] = __float2half(val);
    }

    // Pad to 16 if CHANNELS < 16
    for (int i = CHANNELS + lane_id; i < WMMA_K; i += 32) {
        input_tile[warp_id][0][i] = __float2half(0.0f);
    }
    __syncthreads();

    // Perform WMMA matmul: input [1×16] × weights [16×64] = output [1×64]
    // Need multiple WMMA operations to cover HIDDEN_DIM=64

    const half* weight_ptr = &perception_weights[warp_id * CHANNELS * HIDDEN_DIM];

    for (int out_col = 0; out_col < HIDDEN_DIM; out_col += WMMA_N) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

        wmma::fill_fragment(c_frag, 0.0f);

        // Load input fragment (broadcasted across rows)
        wmma::load_matrix_sync(a_frag, input_tile[warp_id][0], WMMA_K);

        // Load weight fragment
        wmma::load_matrix_sync(b_frag, &weight_ptr[out_col], HIDDEN_DIM);

        // Multiply-accumulate
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        // Store result
        wmma::store_matrix_sync(&output_tile[warp_id][0][out_col], c_frag, WMMA_N, wmma::mem_row_major);
    }

    __syncthreads();

    // Write output
    if (lane_id < HIDDEN_DIM) {
        perception_output[warp_id * grid_cells * HIDDEN_DIM + cell_idx * HIDDEN_DIM + lane_id] =
            output_tile[warp_id][0][lane_id];
    }
}

__global__ void wmma_interaction_kernel(
    const float* perception_output, // [NUM_HEADS × GRID_CELLS_2D × HIDDEN_DIM]
    const half* interaction_weights,// [NUM_HEADS × HIDDEN_DIM × HIDDEN_DIM]
    float* interaction_output,      // [NUM_HEADS × GRID_CELLS_2D × HIDDEN_DIM]
    int grid_cells
) {
    int cell_idx = blockIdx.x;
    if (cell_idx >= grid_cells) return;

    int warp_id = threadIdx.x / 32;
    if (warp_id >= NUM_HEADS) return;

    __shared__ half input_tile[NUM_HEADS][WMMA_M][WMMA_K];
    __shared__ float output_tile[NUM_HEADS][WMMA_M][WMMA_N];

    int lane_id = threadIdx.x % 32;

    // Load perception output and convert to FP16
    if (lane_id < HIDDEN_DIM) {
        float val = perception_output[warp_id * grid_cells * HIDDEN_DIM + cell_idx * HIDDEN_DIM + lane_id];
        input_tile[warp_id][0][lane_id] = __float2half(val);
    }
    __syncthreads();

    // WMMA: [1×64] × [64×64] = [1×64]
    const half* weight_ptr = &interaction_weights[warp_id * HIDDEN_DIM * HIDDEN_DIM];

    for (int out_col = 0; out_col < HIDDEN_DIM; out_col += WMMA_N) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

        wmma::fill_fragment(c_frag, 0.0f);

        wmma::load_matrix_sync(a_frag, input_tile[warp_id][0], WMMA_K);
        wmma::load_matrix_sync(b_frag, &weight_ptr[out_col * HIDDEN_DIM], HIDDEN_DIM);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        wmma::store_matrix_sync(&output_tile[warp_id][0][out_col], c_frag, WMMA_N, wmma::mem_row_major);
    }

    __syncthreads();

    // Apply activation (ReLU) and write output
    if (lane_id < HIDDEN_DIM) {
        float val = output_tile[warp_id][0][lane_id];
        val = fmaxf(val, 0.0f); // ReLU
        interaction_output[warp_id * grid_cells * HIDDEN_DIM + cell_idx * HIDDEN_DIM + lane_id] = val;
    }
}

__global__ void wmma_value_projection_kernel(
    const float* interaction_output, // [NUM_HEADS × GRID_CELLS_2D × HIDDEN_DIM]
    const half* value_weights,       // [NUM_HEADS × HIDDEN_DIM × HEAD_DIM]
    float* ca_output,                // [NUM_HEADS × GRID_CELLS_2D × HEAD_DIM]
    int grid_cells
) {
    int cell_idx = blockIdx.x;
    if (cell_idx >= grid_cells) return;

    int warp_id = threadIdx.x / 32;
    if (warp_id >= NUM_HEADS) return;

    __shared__ half input_tile[NUM_HEADS][WMMA_M][WMMA_K];
    __shared__ float output_tile[NUM_HEADS][WMMA_M][WMMA_N];

    int lane_id = threadIdx.x % 32;

    // Load interaction output
    if (lane_id < HIDDEN_DIM) {
        float val = interaction_output[warp_id * grid_cells * HIDDEN_DIM + cell_idx * HIDDEN_DIM + lane_id];
        input_tile[warp_id][0][lane_id] = __float2half(val);
    }
    __syncthreads();

    // WMMA: [1×64] × [64×16] = [1×16]
    const half* weight_ptr = &value_weights[warp_id * HIDDEN_DIM * HEAD_DIM];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    wmma::load_matrix_sync(a_frag, input_tile[warp_id][0], WMMA_K);
    wmma::load_matrix_sync(b_frag, weight_ptr, HEAD_DIM);

    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    wmma::store_matrix_sync(output_tile[warp_id][0], c_frag, WMMA_N, wmma::mem_row_major);

    __syncthreads();

    // Write final output
    if (lane_id < HEAD_DIM) {
        ca_output[warp_id * grid_cells * HEAD_DIM + cell_idx * HEAD_DIM + lane_id] =
            output_tile[warp_id][0][lane_id];
    }
}

#else
// Fallback for architectures without Tensor Cores (CC < 7.0)
__global__ void fallback_matmul_kernel() {
    // Empty fallback - will implement FP32 version if needed
}
#endif
