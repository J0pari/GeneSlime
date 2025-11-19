#include <cuda_runtime.h>
#include "../core/types.cuh"
#include "../utils/tile_ops.cuh"
#include "../kernels/svd_jacobi.cu"

// ============================================================================
// Coherence Computation (∂(prediction_error)/∂t)
// ============================================================================

__global__ void compute_coherence_kernel(
    const float* prediction_errors,  // [CORRELATION_WINDOW]
    float* coherence,
    int window_size
) {
    __shared__ float errors[CORRELATION_WINDOW];

    int tid = threadIdx.x;

    // Load errors into shared memory
    if (tid < window_size) {
        errors[tid] = prediction_errors[tid];
    }
    __syncthreads();

    // Compute temporal derivative using central differences
    float derivative_sum = 0.0f;
    if (tid > 0 && tid < window_size - 1) {
        float derivative = (errors[tid + 1] - errors[tid - 1]) / 2.0f;
        derivative_sum = derivative;
    }

    // Reduce to compute mean absolute derivative
    derivative_sum = BlockReduce<BLOCK_SIZE_1D>::sum(fabsf(derivative_sum));

    if (tid == 0) {
        *coherence = 1.0f / (1.0f + derivative_sum / window_size); // Inverse derivative = stability
    }
}

// ============================================================================
// Prediction Error (Simple Autoregressive Model)
// ============================================================================

__global__ void compute_prediction_error_kernel(
    const float* ca_concentration_history,  // [CORRELATION_WINDOW × GRID_CELLS_2D × CHANNELS]
    float* prediction_errors,                // [CORRELATION_WINDOW]
    int window_size,
    int grid_cells
) {
    int t = blockIdx.x;
    if (t >= window_size - 1) return;  // Need t+1 for prediction

    int tid = threadIdx.x;

    // Simple prediction: A^(t+1) = A^t (persistence model)
    float error = 0.0f;
    for (int i = tid; i < grid_cells * CHANNELS; i += blockDim.x) {
        float predicted = ca_concentration_history[t * grid_cells * CHANNELS + i];
        float actual = ca_concentration_history[(t + 1) * grid_cells * CHANNELS + i];
        float diff = predicted - actual;
        error += diff * diff;
    }

    // Reduce error
    error = BlockReduce<BLOCK_SIZE_1D>::sum(error);

    if (tid == 0) {
        prediction_errors[t] = sqrtf(error / (grid_cells * CHANNELS));
    }
}

// ============================================================================
// Causal Attribution (Segment Contribution to Fitness)
// ============================================================================

__global__ void compute_causal_attribution_kernel(
    Organism* organism,
    const float* segment_activation_history,  // [CORRELATION_WINDOW × NUM_SEGMENTS]
    const float* fitness_history,             // [CORRELATION_WINDOW]
    int window_size
) {
    int seg_idx = threadIdx.x;
    if (seg_idx >= NUM_SEGMENTS) return;

    // Compute correlation between segment activation and fitness
    float mean_activation = 0.0f;
    float mean_fitness = 0.0f;

    for (int t = 0; t < window_size; t++) {
        mean_activation += segment_activation_history[t * NUM_SEGMENTS + seg_idx];
        mean_fitness += fitness_history[t];
    }
    mean_activation /= window_size;
    mean_fitness /= window_size;

    float cov = 0.0f;
    float var_activation = 0.0f;
    float var_fitness = 0.0f;

    for (int t = 0; t < window_size; t++) {
        float da = segment_activation_history[t * NUM_SEGMENTS + seg_idx] - mean_activation;
        float df = fitness_history[t] - mean_fitness;

        cov += da * df;
        var_activation += da * da;
        var_fitness += df * df;
    }

    float correlation = cov / (sqrtf(var_activation * var_fitness) + 1e-10f);

    // Store causal attribution
    organism->genome.segments[seg_idx].causal_attribution = fabsf(correlation);
}

// ============================================================================
// Combined Fitness: effective_rank × coherence
// ============================================================================

__global__ void compute_fitness_kernel(
    Organism* population,
    const float* segment_correlation_matrices,  // [POPULATION_SIZE × NUM_SEGMENTS × NUM_SEGMENTS]
    const float* prediction_error_histories,    // [POPULATION_SIZE × CORRELATION_WINDOW]
    int pop_size
) {
    int idx = blockIdx.x;
    if (idx >= pop_size) return;

    Organism* org = &population[idx];

    extern __shared__ float smem[];
    float* corr_matrix = smem;
    float* singular_values = &smem[NUM_SEGMENTS * NUM_SEGMENTS];
    float* pred_errors = &smem[NUM_SEGMENTS * NUM_SEGMENTS + NUM_SEGMENTS];

    int tid = threadIdx.x;

    // Load correlation matrix
    const float* org_corr = &segment_correlation_matrices[idx * NUM_SEGMENTS * NUM_SEGMENTS];
    for (int i = tid; i < NUM_SEGMENTS * NUM_SEGMENTS; i += blockDim.x) {
        corr_matrix[i] = org_corr[i];
    }

    // Load prediction errors
    const float* org_errors = &prediction_error_histories[idx * CORRELATION_WINDOW];
    if (tid < CORRELATION_WINDOW) {
        pred_errors[tid] = org_errors[tid];
    }
    __syncthreads();

    // Compute effective rank via SVD
    if (tid < WARP_SIZE) {
        warp_svd_jacobi(corr_matrix, singular_values, NUM_SEGMENTS);
    }
    __syncthreads();

    // Compute entropy of singular values
    float sum_sv = 0.0f;
    if (tid < NUM_SEGMENTS) {
        sum_sv = singular_values[tid];
    }
    sum_sv = BlockReduce<BLOCK_SIZE_1D>::sum(sum_sv);
    __syncthreads();

    float entropy = 0.0f;
    if (tid < NUM_SEGMENTS) {
        float p = singular_values[tid] / (sum_sv + 1e-10f);
        if (p > 1e-10f) {
            entropy -= p * log2f(p);
        }
    }
    entropy = BlockReduce<BLOCK_SIZE_1D>::sum(entropy);

    float effective_rank = exp2f(entropy);

    // Compute coherence (temporal derivative of prediction error)
    float derivative_sum = 0.0f;
    if (tid > 0 && tid < CORRELATION_WINDOW - 1) {
        float deriv = (pred_errors[tid + 1] - pred_errors[tid - 1]) / 2.0f;
        derivative_sum = fabsf(deriv);
    }
    derivative_sum = BlockReduce<BLOCK_SIZE_1D>::sum(derivative_sum);

    float coherence = 1.0f / (1.0f + derivative_sum / CORRELATION_WINDOW);

    // Combined fitness
    if (tid == 0) {
        org->effective_rank = effective_rank;
        org->coherence = coherence;
        org->fitness = effective_rank * coherence;
    }
}
