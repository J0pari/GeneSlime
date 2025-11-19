#include <cuda_runtime.h>
#include "../core/types.cuh"
#include "../utils/tile_ops.cuh"

// ============================================================================
// Warp-Level SVD using Jacobi Iterations
// ============================================================================

__device__ void jacobi_rotation_2x2(
    float& a11, float& a12, float& a22,
    float& c, float& s
) {
    // Compute rotation angle for 2×2 submatrix
    if (fabsf(a12) < 1e-10f) {
        c = 1.0f;
        s = 0.0f;
        return;
    }

    float tau = (a22 - a11) / (2.0f * a12);
    float t = (tau >= 0.0f) ? 1.0f / (tau + sqrtf(1.0f + tau * tau))
                            : -1.0f / (-tau + sqrtf(1.0f + tau * tau));
    c = 1.0f / sqrtf(1.0f + t * t);
    s = t * c;
}

__device__ void warp_svd_jacobi(
    float* A,          // [N×N] symmetric matrix (shared memory)
    float* S,          // [N] singular values output
    int N,
    int max_iterations = SVD_MAX_ITERATIONS
) {
    int lane = threadIdx.x % WARP_SIZE;

    // Jacobi iterations
    for (int iter = 0; iter < max_iterations; iter++) {
        float max_off_diag = 0.0f;

        // Sweep over all off-diagonal elements
        for (int p = 0; p < N - 1; p++) {
            for (int q = p + 1; q < N; q++) {
                float a_pp = A[p * N + p];
                float a_qq = A[q * N + q];
                float a_pq = A[p * N + q];

                // Check convergence
                if (fabsf(a_pq) < SVD_TOLERANCE) continue;
                max_off_diag = fmaxf(max_off_diag, fabsf(a_pq));

                // Compute rotation
                float c, s;
                jacobi_rotation_2x2(a_pp, a_pq, a_qq, c, s);

                // Apply rotation to matrix (each thread handles one row)
                if (lane < N) {
                    int i = lane;

                    // Update row i
                    float a_ip = A[i * N + p];
                    float a_iq = A[i * N + q];

                    A[i * N + p] = c * a_ip - s * a_iq;
                    A[i * N + q] = s * a_ip + c * a_iq;

                    // Update column i (symmetric)
                    float a_pi = A[p * N + i];
                    float a_qi = A[q * N + i];

                    A[p * N + i] = c * a_pi - s * a_qi;
                    A[q * N + i] = s * a_pi + c * a_qi;
                }

                // Update diagonal elements
                if (lane == 0) {
                    A[p * N + p] = c * c * a_pp - 2.0f * c * s * a_pq + s * s * a_qq;
                    A[q * N + q] = s * s * a_pp + 2.0f * c * s * a_pq + c * c * a_qq;
                    A[p * N + q] = 0.0f;
                    A[q * N + p] = 0.0f;
                }
            }
        }

        // Check global convergence
        __syncwarp();
        if (max_off_diag < SVD_TOLERANCE) break;
    }

    // Extract singular values (diagonal elements)
    if (lane < N) {
        S[lane] = sqrtf(fabsf(A[lane * N + lane]));
    }
}

// ============================================================================
// Segment Correlation Matrix
// ============================================================================

__global__ void compute_segment_correlation_kernel(
    const Organism* organism,
    float* correlation_matrix,  // [NUM_SEGMENTS × NUM_SEGMENTS]
    const float* history,        // [CORRELATION_WINDOW × NUM_SEGMENTS]
    int window_size
) {
    int seg_i = blockIdx.x;
    int seg_j = blockIdx.y;

    if (seg_i >= NUM_SEGMENTS || seg_j >= NUM_SEGMENTS) return;

    // Compute correlation between segment i and segment j over time window
    float mean_i = 0.0f;
    float mean_j = 0.0f;

    for (int t = 0; t < window_size; t++) {
        mean_i += history[t * NUM_SEGMENTS + seg_i];
        mean_j += history[t * NUM_SEGMENTS + seg_j];
    }
    mean_i /= window_size;
    mean_j /= window_size;

    float cov = 0.0f;
    float var_i = 0.0f;
    float var_j = 0.0f;

    for (int t = 0; t < window_size; t++) {
        float di = history[t * NUM_SEGMENTS + seg_i] - mean_i;
        float dj = history[t * NUM_SEGMENTS + seg_j] - mean_j;

        cov += di * dj;
        var_i += di * di;
        var_j += dj * dj;
    }

    float corr = cov / (sqrtf(var_i * var_j) + 1e-10f);
    correlation_matrix[seg_i * NUM_SEGMENTS + seg_j] = corr;
}

// ============================================================================
// Effective Rank Computation
// ============================================================================

__global__ void compute_effective_rank_kernel(
    const float* correlation_matrix,  // [NUM_SEGMENTS × NUM_SEGMENTS]
    float* effective_rank,
    int n_segments
) {
    extern __shared__ float smem[];
    float* local_matrix = smem;
    float* singular_values = &smem[n_segments * n_segments];

    int tid = threadIdx.x;

    // Load correlation matrix into shared memory
    for (int i = tid; i < n_segments * n_segments; i += blockDim.x) {
        local_matrix[i] = correlation_matrix[i];
    }
    __syncthreads();

    // Perform Jacobi SVD (warp 0 only)
    if (tid < WARP_SIZE) {
        warp_svd_jacobi(local_matrix, singular_values, n_segments);
    }
    __syncthreads();

    // Compute effective rank: entropy of normalized singular values
    float sum_sv = 0.0f;
    if (tid < n_segments) {
        sum_sv = singular_values[tid];
    }

    sum_sv = BlockReduce<BLOCK_SIZE_1D>::sum(sum_sv);
    __syncthreads();

    float entropy = 0.0f;
    if (tid < n_segments) {
        float p = singular_values[tid] / (sum_sv + 1e-10f);
        if (p > 1e-10f) {
            entropy -= p * log2f(p);
        }
    }

    entropy = BlockReduce<BLOCK_SIZE_1D>::sum(entropy);

    if (tid == 0) {
        *effective_rank = exp2f(entropy); // Convert entropy to effective rank
    }
}
