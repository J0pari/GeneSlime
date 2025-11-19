#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "types.cuh"
#include "../utils/tile_ops.cuh"
#include "../config/constants.cuh"
#include "../kernels/utils.cu"

// ============================================================================
// Flow-Lenia: Mass-Conservative Continuous Cellular Automata
// Based on Plantec et al. (2025) - arXiv:2506.08569v1
//
// Key equations:
// - Kernel K_i(x) = Σ_j b_{i,j} exp(-(||x||/(r_i·R) - a_{i,j})² / (2w²_{i,j}))
// - Growth G_i(x) = 2 exp(-(μ_i - x)² / (2σ²_i)) - 1
// - Affinity U^t_j = Σ_i h_i · G_i(K_i * A^t_{c0_i}) · [c1_i == j]
// - Flow F^t_i = (1-α^t)∇U^t_i - α^t∇A^t_Σ
// - Alpha α^t(x) = [(A^t_Σ(x)/β_A)^n]^1_0
// - Reintegration A^{t+dt}_i(x) = Σ_{x'} A^t_i(x') I_i(x', x)
// ============================================================================

// ============================================================================
// Kernel & Growth Function Definitions
// ============================================================================

// Growth kernel: Sum of k concentric Gaussian rings (Eq. 1)
__device__ float compute_growth_kernel(
    float distance,
    const float* a_rings,    // [k] ring positions in [0,1]
    const float* b_amplitudes, // [k] ring amplitudes
    const float* w_widths,   // [k] ring widths
    float r_scale,           // kernel radius scale in [0,1]
    float R_max,             // maximum neighborhood radius
    int k                    // number of rings (typically 3)
) {
    float normalized_dist = distance / (r_scale * R_max);
    float value = 0.0f;

    #pragma unroll
    for (int j = 0; j < k; j++) {
        float delta = normalized_dist - a_rings[j];
        float w_sq = w_widths[j] * w_widths[j];
        value += b_amplitudes[j] * expf(-(delta * delta) / (2.0f * w_sq));
    }

    return value;
}

// Growth function: Scaled Gaussian in [-1, 1] (Eq. 2)
__device__ __forceinline__ float growth_function(float x, float mu, float sigma) {
    float delta = mu - x;
    float sigma_sq = sigma * sigma;
    return 2.0f * expf(-(delta * delta) / (2.0f * sigma_sq)) - 1.0f;
}

// ============================================================================
// Stage 1: Compute Affinity Map U^t (Eq. 3)
// This is the "growth" term in Lenia, reinterpreted as affinity
// U^t_j = Σ_i h_i · G_i(K_i * A^t_{c0_i}) · [c1_i == j]
// ============================================================================

struct KernelSpec {
    float a[3];      // Ring positions
    float b[3];      // Ring amplitudes
    float w[3];      // Ring widths
    float r;         // Radius scale
    float mu;        // Growth function mean
    float sigma;     // Growth function std
    int c_source;    // Source channel
    int c_target;    // Target channel
    float h_weight;  // Kernel importance weight
};

__global__ void compute_affinity_map_kernel(
    const float* ca_concentration,  // A^t [GRID_SIZE² × CHANNELS]
    const KernelSpec* kernels,       // [num_kernels] kernel specifications
    float* affinity_map,             // U^t [GRID_SIZE² × CHANNELS] output
    int num_kernels,
    float R_max
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= GRID_SIZE || y >= GRID_SIZE) return;

    int cell_idx = y * GRID_SIZE + x;

    // Zero initialize affinity
    for (int c = 0; c < CHANNELS; c++) {
        affinity_map[cell_idx * CHANNELS + c] = 0.0f;
    }

    // Compute contribution from each kernel
    for (int k_idx = 0; k_idx < num_kernels; k_idx++) {
        const KernelSpec& kernel = kernels[k_idx];

        // Convolve kernel with source channel in neighborhood
        float convolution = 0.0f;
        float kernel_normalization = 0.0f;

        int radius = (int)ceilf(kernel.r * R_max);

        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                // Periodic boundary conditions
                int nx = (x + dx + GRID_SIZE) % GRID_SIZE;
                int ny = (y + dy + GRID_SIZE) % GRID_SIZE;

                float dist = sqrtf((float)(dx * dx + dy * dy));

                // Evaluate kernel at this distance
                float k_val = compute_growth_kernel(
                    dist,
                    kernel.a,
                    kernel.b,
                    kernel.w,
                    kernel.r,
                    R_max,
                    3  // k=3 rings
                );

                int neighbor_idx = ny * GRID_SIZE + nx;
                float concentration = ca_concentration[neighbor_idx * CHANNELS + kernel.c_source];

                convolution += k_val * concentration;
                kernel_normalization += k_val;
            }
        }

        // Normalize convolution (kernels satisfy ∫K=1 but discretization needs correction)
        if (kernel_normalization > 1e-8f) {
            convolution /= kernel_normalization;
        }

        // Apply growth function G_i
        float growth = growth_function(convolution, kernel.mu, kernel.sigma);

        // Accumulate weighted contribution to target channel affinity
        atomicAdd(&affinity_map[cell_idx * CHANNELS + kernel.c_target],
                  kernel.h_weight * growth);
    }
}

// ============================================================================
// Stage 2: Compute Flow Field F^t (Eq. 5)
// F^t_i = (1-α^t)∇U^t_i - α^t∇A^t_Σ
// α^t(x) = [(A^t_Σ(x)/β_A)^n]^1_0  clamped to [0,1]
// ============================================================================

__global__ void compute_flow_field_kernel(
    const float* affinity_map,      // U^t [GRID_SIZE² × CHANNELS]
    const float* ca_concentration,  // A^t [GRID_SIZE² × CHANNELS]
    float* flow_field_x,            // F^t_x [GRID_SIZE² × CHANNELS] output
    float* flow_field_y,            // F^t_y [GRID_SIZE² × CHANNELS] output
    float beta_A,                    // Critical mass threshold
    int n_power                      // Power for alpha (typically 2)
) {
    // Use tiled loading with halo for efficient gradient computation
    using Tile = TiledSection2D<BLOCK_SIZE_2D, 1, 1>;

    __shared__ float tile_mass[Tile::PADDED][Tile::PADDED];

    int x = blockIdx.x * BLOCK_SIZE_2D + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE_2D + threadIdx.y;

    if (x >= GRID_SIZE || y >= GRID_SIZE) return;

    int cell_idx = y * GRID_SIZE + x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Compute total mass A^t_Σ(x) = Σ_c A^t_c(x)
    float total_mass = 0.0f;
    for (int c = 0; c < CHANNELS; c++) {
        total_mass += ca_concentration[cell_idx * CHANNELS + c];
    }

    // Compute α^t(x) = [(A^t_Σ(x)/β_A)^n]^1_0
    float alpha = powf(total_mass / beta_A, (float)n_power);
    alpha = clamp(alpha, 0.0f, 1.0f);

    // Load total mass into shared memory tile with halo
    tile_mass[ty + 1][tx + 1] = total_mass;

    // Load halo regions (8-connected neighborhood)
    if (tx == 0 && x > 0) {
        float mass = 0.0f;
        int idx = y * GRID_SIZE + (x - 1);
        for (int c = 0; c < CHANNELS; c++) {
            mass += ca_concentration[idx * CHANNELS + c];
        }
        tile_mass[ty + 1][0] = mass;
    }
    if (tx == BLOCK_SIZE_2D - 1 && x < GRID_SIZE - 1) {
        float mass = 0.0f;
        int idx = y * GRID_SIZE + (x + 1);
        for (int c = 0; c < CHANNELS; c++) {
            mass += ca_concentration[idx * CHANNELS + c];
        }
        tile_mass[ty + 1][tx + 2] = mass;
    }
    if (ty == 0 && y > 0) {
        float mass = 0.0f;
        int idx = (y - 1) * GRID_SIZE + x;
        for (int c = 0; c < CHANNELS; c++) {
            mass += ca_concentration[idx * CHANNELS + c];
        }
        tile_mass[0][tx + 1] = mass;
    }
    if (ty == BLOCK_SIZE_2D - 1 && y < GRID_SIZE - 1) {
        float mass = 0.0f;
        int idx = (y + 1) * GRID_SIZE + x;
        for (int c = 0; c < CHANNELS; c++) {
            mass += ca_concentration[idx * CHANNELS + c];
        }
        tile_mass[ty + 2][tx + 1] = mass;
    }

    __syncthreads();

    // Compute ∇A^t_Σ using Sobel filter (as per paper: "gradients estimated through Sobel filtering")
    // Sobel X: [[-1,0,1], [-2,0,2], [-1,0,1]] / 8
    // Sobel Y: [[-1,-2,-1], [0,0,0], [1,2,1]] / 8
    float grad_mass_x = (-tile_mass[ty][tx] + tile_mass[ty][tx+2]
                        -2*tile_mass[ty+1][tx] + 2*tile_mass[ty+1][tx+2]
                        -tile_mass[ty+2][tx] + tile_mass[ty+2][tx+2]) / 8.0f;

    float grad_mass_y = (-tile_mass[ty][tx] - 2*tile_mass[ty][tx+1] - tile_mass[ty][tx+2]
                        +tile_mass[ty+2][tx] + 2*tile_mass[ty+2][tx+1] + tile_mass[ty+2][tx+2]) / 8.0f;

    // Compute flow for each channel
    for (int c = 0; c < CHANNELS; c++) {
        // Build affinity stencil for Sobel gradient
        float affinity_stencil[3][3];

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = clamp(x + dx, 0, GRID_SIZE - 1);
                int ny = clamp(y + dy, 0, GRID_SIZE - 1);
                affinity_stencil[dy+1][dx+1] = affinity_map[(ny * GRID_SIZE + nx) * CHANNELS + c];
            }
        }

        // Compute ∇U^t_c using Sobel
        float grad_affinity_x = Stencils::gradient_x(affinity_stencil) * 4.0f; // Stencils uses /2, we want /8
        float grad_affinity_y = Stencils::gradient_y(affinity_stencil) * 4.0f;

        // F^t_c = (1-α^t)∇U^t_c - α^t∇A^t_Σ (Eq. 5)
        flow_field_x[cell_idx * CHANNELS + c] = (1.0f - alpha) * grad_affinity_x - alpha * grad_mass_x;
        flow_field_y[cell_idx * CHANNELS + c] = (1.0f - alpha) * grad_affinity_y - alpha * grad_mass_y;
    }
}

// ============================================================================
// Stage 3: Reintegration Tracking (Eq. 6)
// Mass-conserving transport via overlap integrals
// A^{t+dt}_i(x) = Σ_{x'∈L} A^t_i(x') I_i(x', x)
// I_i(x', x) = ∫_{Ω(x)} D(x''_i, s) where x''_i = x' + dt·F^t_i(x')
// D is uniform square distribution with side 2s
// ============================================================================

__device__ float square_overlap_integral(
    float center_x,
    float center_y,
    float dist_half_side,  // s (half side length)
    float cell_x,
    float cell_y
) {
    // Compute overlap of square distribution D centered at (center_x, center_y)
    // with side length 2*dist_half_side, with unit cell centered at (cell_x, cell_y)

    float cell_half = 0.5f;

    // Distribution bounds
    float dist_x_min = center_x - dist_half_side;
    float dist_x_max = center_x + dist_half_side;
    float dist_y_min = center_y - dist_half_side;
    float dist_y_max = center_y + dist_half_side;

    // Cell bounds
    float cell_x_min = cell_x - cell_half;
    float cell_x_max = cell_x + cell_half;
    float cell_y_min = cell_y - cell_half;
    float cell_y_max = cell_y + cell_half;

    // Compute overlap in each dimension
    float overlap_x = fmaxf(0.0f, fminf(dist_x_max, cell_x_max) - fmaxf(dist_x_min, cell_x_min));
    float overlap_y = fmaxf(0.0f, fminf(dist_y_max, cell_y_max) - fmaxf(dist_y_min, cell_y_min));

    float overlap_area = overlap_x * overlap_y;
    float dist_total_area = (2.0f * dist_half_side) * (2.0f * dist_half_side);

    // Return fraction of distribution mass in this cell (ensures Σ I = 1)
    return (dist_total_area > 1e-10f) ? (overlap_area / dist_total_area) : 0.0f;
}

__global__ void reintegration_tracking_kernel(
    const float* ca_concentration,  // A^t [GRID_SIZE² × CHANNELS]
    const float* flow_field_x,      // F^t_x [GRID_SIZE² × CHANNELS]
    const float* flow_field_y,      // F^t_y [GRID_SIZE² × CHANNELS]
    float* next_concentration,      // A^{t+dt} [GRID_SIZE² × CHANNELS] output
    float dt,
    float s  // Distribution spread (temperature/diffusion parameter)
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= GRID_SIZE || y >= GRID_SIZE) return;

    int cell_idx = y * GRID_SIZE + x;

    // Initialize output to zero
    for (int c = 0; c < CHANNELS; c++) {
        next_concentration[cell_idx * CHANNELS + c] = 0.0f;
    }

    // Gather mass from neighborhood (Chebyshev distance ≤ 5 as per paper)
    constexpr int SEARCH_RADIUS = 5;

    for (int source_dy = -SEARCH_RADIUS; source_dy <= SEARCH_RADIUS; source_dy++) {
        for (int source_dx = -SEARCH_RADIUS; source_dx <= SEARCH_RADIUS; source_dx++) {
            // Source cell with periodic boundaries
            int source_x = (x + source_dx + GRID_SIZE) % GRID_SIZE;
            int source_y = (y + source_dy + GRID_SIZE) % GRID_SIZE;
            int source_idx = source_y * GRID_SIZE + source_x;

            for (int c = 0; c < CHANNELS; c++) {
                float mass = ca_concentration[source_idx * CHANNELS + c];
                if (mass < 1e-10f) continue;  // Skip negligible mass

                // Compute target position x'' = x' + dt·F^t(x')
                float fx = flow_field_x[source_idx * CHANNELS + c];
                float fy = flow_field_y[source_idx * CHANNELS + c];

                float target_x = source_x + dt * fx;
                float target_y = source_y + dt * fy;

                // Compute overlap integral I_c(source, target)
                float overlap = square_overlap_integral(
                    target_x, target_y,
                    s,  // Distribution half-side
                    (float)x, (float)y  // Target cell center
                );

                // Accumulate incoming mass (atomics ensure thread safety)
                if (overlap > 1e-10f) {
                    atomicAdd(&next_concentration[cell_idx * CHANNELS + c], mass * overlap);
                }
            }
        }
    }
}

// ============================================================================
// Mass Conservation Verification
// Orchestrates calls to compute_total_mass_kernel from utils.cu
// ============================================================================

__host__ float verify_mass_conservation(
    const float* ca_prev,
    const float* ca_next,
    int grid_cells,
    int channels
) {
    float *d_mass_prev, *d_mass_next;
    cudaMalloc(&d_mass_prev, sizeof(float));
    cudaMalloc(&d_mass_next, sizeof(float));
    cudaMemset(d_mass_prev, 0, sizeof(float));
    cudaMemset(d_mass_next, 0, sizeof(float));

    // Use existing utility from kernels/utils.cu
    compute_total_mass_kernel<<<32, BLOCK_SIZE_1D>>>(ca_prev, d_mass_prev, grid_cells, channels);
    compute_total_mass_kernel<<<32, BLOCK_SIZE_1D>>>(ca_next, d_mass_next, grid_cells, channels);

    float mass_prev, mass_next;
    cudaMemcpy(&mass_prev, d_mass_prev, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&mass_next, d_mass_next, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_mass_prev);
    cudaFree(d_mass_next);

    return fabsf(mass_next - mass_prev);
}

// ============================================================================
// Helper: Initialize kernels from genome parameters
// ============================================================================

__global__ void initialize_kernels_from_genome_kernel(
    const Genome* genome,
    KernelSpec* kernels,
    int num_kernels
) {
    using GridStride = GridStride1D;

    for (int idx = GridStride::start(); idx < num_kernels; idx += GridStride::stride()) {

        // Extract kernel parameters from genome segments
        // This maps genome representation to kernel specifications
        int seg_idx = idx % NUM_SEGMENTS;

        if (genome->is_active(seg_idx)) {
            const GenomeSegment& seg = genome->segments[seg_idx];
            KernelSpec& kernel = kernels[idx];

            // Map payload to kernel parameters
            kernel.a[0] = genome_to_param(seg.payload[0], 0.0f, 1.0f);
            kernel.a[1] = genome_to_param(seg.payload[1], 0.0f, 1.0f);
            kernel.a[2] = genome_to_param(seg.payload[2], 0.0f, 1.0f);

            kernel.b[0] = genome_to_param(seg.payload[3], 0.0f, 1.0f);
            kernel.b[1] = genome_to_param(seg.payload[4], 0.0f, 1.0f);
            kernel.b[2] = genome_to_param(seg.payload[5], 0.0f, 1.0f);

            kernel.w[0] = genome_to_param(seg.payload[6], 0.01f, 0.5f);
            kernel.w[1] = genome_to_param(seg.payload[7], 0.01f, 0.5f);
            kernel.w[2] = genome_to_param(seg.payload[8], 0.01f, 0.5f);

            kernel.r = genome_to_param(seg.payload[9], 0.2f, 1.0f);
            kernel.mu = genome_to_param(seg.payload[10], 0.05f, 0.5f);
            kernel.sigma = genome_to_param(seg.payload[11], 0.001f, 0.2f);

            kernel.c_source = seg.get_grid_x() % CHANNELS;
            kernel.c_target = seg.get_grid_y() % CHANNELS;
            kernel.h_weight = genome_to_param(seg.payload[12], 0.0f, 1.0f);
        }
    }
}
