#include <cuda_runtime.h>
#include "types.cuh"
#include "../utils/tile_ops.cuh"

// ============================================================================
// Multi-Head CA Forward Pass (FP32 Fallback)
// ============================================================================

__global__ void multihead_ca_fp32_kernel(
    const float* ca_concentration,  // [GRID_CELLS_2D × CHANNELS]
    const half* perception_weights, // [NUM_HEADS × CHANNELS × HIDDEN_DIM]
    const half* interaction_weights,// [NUM_HEADS × HIDDEN_DIM × HIDDEN_DIM]
    const half* value_weights,      // [NUM_HEADS × HIDDEN_DIM × HEAD_DIM]
    float* ca_output,               // [NUM_HEADS × GRID_CELLS_2D × HEAD_DIM]
    int grid_cells
) {
    int cell_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    if (cell_idx >= grid_cells || head_idx >= NUM_HEADS) return;

    extern __shared__ float smem[];
    float* perception_out = smem;                          // [HIDDEN_DIM]
    float* interaction_out = &smem[HIDDEN_DIM];           // [HIDDEN_DIM]

    int tid = threadIdx.x;

    // Step 1: Perception (CHANNELS → HIDDEN_DIM)
    if (tid < HIDDEN_DIM) {
        float sum = 0.0f;
        for (int c = 0; c < CHANNELS; c++) {
            float input = ca_concentration[cell_idx * CHANNELS + c];
            float weight = __half2float(perception_weights[head_idx * CHANNELS * HIDDEN_DIM + c * HIDDEN_DIM + tid]);
            sum += input * weight;
        }
        perception_out[tid] = sum;
    }
    __syncthreads();

    // Step 2: Interaction (HIDDEN_DIM → HIDDEN_DIM)
    if (tid < HIDDEN_DIM) {
        float sum = 0.0f;
        for (int h = 0; h < HIDDEN_DIM; h++) {
            float input = perception_out[h];
            float weight = __half2float(interaction_weights[head_idx * HIDDEN_DIM * HIDDEN_DIM + h * HIDDEN_DIM + tid]);
            sum += input * weight;
        }
        interaction_out[tid] = fmaxf(sum, 0.0f); // ReLU
    }
    __syncthreads();

    // Step 3: Value Projection (HIDDEN_DIM → HEAD_DIM)
    if (tid < HEAD_DIM) {
        float sum = 0.0f;
        for (int h = 0; h < HIDDEN_DIM; h++) {
            float input = interaction_out[h];
            float weight = __half2float(value_weights[head_idx * HIDDEN_DIM * HEAD_DIM + h * HEAD_DIM + tid]);
            sum += input * weight;
        }
        ca_output[head_idx * grid_cells * HEAD_DIM + cell_idx * HEAD_DIM + tid] = sum;
    }
}

// ============================================================================
// Multi-Head CA with Neighborhood Aggregation
// ============================================================================

__global__ void multihead_ca_with_neighbors_kernel(
    const float* ca_concentration,  // [GRID_CELLS_2D × CHANNELS]
    const half* perception_weights,
    const half* interaction_weights,
    const half* value_weights,
    float* ca_output,
    int grid_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int head_idx = blockIdx.z;

    if (x >= grid_size || y >= grid_size || head_idx >= NUM_HEADS) return;

    __shared__ float tile[BLOCK_SIZE_2D + 2][BLOCK_SIZE_2D + 2][CHANNELS];

    // Load tile with halo
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Center
    if (x < grid_size && y < grid_size) {
        for (int c = 0; c < CHANNELS; c++) {
            tile[ty + 1][tx + 1][c] = ca_concentration[(y * grid_size + x) * CHANNELS + c];
        }
    }

    // Halo regions (8-connected neighborhood)
    if (tx == 0 && x > 0) {
        for (int c = 0; c < CHANNELS; c++) {
            tile[ty + 1][0][c] = ca_concentration[(y * grid_size + (x - 1)) * CHANNELS + c];
        }
    }
    if (tx == BLOCK_SIZE_2D - 1 && x < grid_size - 1) {
        for (int c = 0; c < CHANNELS; c++) {
            tile[ty + 1][BLOCK_SIZE_2D + 1][c] = ca_concentration[(y * grid_size + (x + 1)) * CHANNELS + c];
        }
    }
    if (ty == 0 && y > 0) {
        for (int c = 0; c < CHANNELS; c++) {
            tile[0][tx + 1][c] = ca_concentration[((y - 1) * grid_size + x) * CHANNELS + c];
        }
    }
    if (ty == BLOCK_SIZE_2D - 1 && y < grid_size - 1) {
        for (int c = 0; c < CHANNELS; c++) {
            tile[BLOCK_SIZE_2D + 1][tx + 1][c] = ca_concentration[((y + 1) * grid_size + x) * CHANNELS + c];
        }
    }

    __syncthreads();

    // Aggregate neighborhood (9-point stencil)
    float neighborhood[CHANNELS];
    for (int c = 0; c < CHANNELS; c++) {
        neighborhood[c] = 0.0f;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                neighborhood[c] += tile[ty + 1 + dy][tx + 1 + dx][c];
            }
        }
        neighborhood[c] /= 9.0f; // Average
    }

    // Perception: CHANNELS → HIDDEN_DIM
    float perception_out[HIDDEN_DIM];
    for (int h = 0; h < HIDDEN_DIM; h++) {
        float sum = 0.0f;
        for (int c = 0; c < CHANNELS; c++) {
            float weight = __half2float(perception_weights[head_idx * CHANNELS * HIDDEN_DIM + c * HIDDEN_DIM + h]);
            sum += neighborhood[c] * weight;
        }
        perception_out[h] = sum;
    }

    // Interaction: HIDDEN_DIM → HIDDEN_DIM
    float interaction_out[HIDDEN_DIM];
    for (int h = 0; h < HIDDEN_DIM; h++) {
        float sum = 0.0f;
        for (int h2 = 0; h2 < HIDDEN_DIM; h2++) {
            float weight = __half2float(interaction_weights[head_idx * HIDDEN_DIM * HIDDEN_DIM + h2 * HIDDEN_DIM + h]);
            sum += perception_out[h2] * weight;
        }
        interaction_out[h] = fmaxf(sum, 0.0f); // ReLU
    }

    // Value projection: HIDDEN_DIM → HEAD_DIM
    int cell_idx = y * grid_size + x;
    for (int d = 0; d < HEAD_DIM; d++) {
        float sum = 0.0f;
        for (int h = 0; h < HIDDEN_DIM; h++) {
            float weight = __half2float(value_weights[head_idx * HIDDEN_DIM * HEAD_DIM + h * HEAD_DIM + d]);
            sum += interaction_out[h] * weight;
        }
        ca_output[head_idx * grid_size * grid_size * HEAD_DIM + cell_idx * HEAD_DIM + d] = sum;
    }
}
