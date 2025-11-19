#include "../config/constants.cuh"
#include "../core/types.cuh"
#include "../kernels/utils.cu"
#include "../kernels/svd_jacobi.cu"
#include "../utils/tile_ops.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>

// MAP-Elites archive with Voronoi cells in behavioral space
// Uses forward genome compression and hash-based deduplication
// Behavioral embedding via DIRESA (differential reservoir sampling)

struct CompressedGenome {
    float* U_matrix;           // Left singular vectors (NUM_SEGMENTS × RANK)
    float* singular_values;    // Diagonal (RANK)
    float* Vt_matrix;          // Right singular vectors (RANK × SEGMENT_PAYLOAD_SIZE)
    int rank;                  // Effective rank
    uint64_t hash;             // Full genome hash for deduplication
    uint8_t permission_fingerprint[NUM_SEGMENTS];  // Quick permission check
};

struct VoronoiCell {
    float centroid[BEHAVIOR_DIM];
    CompressedGenome elite_genome;
    float elite_fitness;
    float elite_behavior[BEHAVIOR_DIM];
    uint32_t occupation_count;
    uint32_t last_update_generation;
    bool occupied;
};

struct Archive {
    VoronoiCell* cells;
    int num_cells;
    float* behavioral_bounds_min;  // [BEHAVIOR_DIM]
    float* behavioral_bounds_max;  // [BEHAVIOR_DIM]
    uint32_t total_insertions;
    uint32_t successful_insertions;
};

// DIRESA: Differential reservoir sampling for behavioral embedding
// Extracts low-dimensional behavioral coordinates from high-dimensional CA states
__global__ void compute_behavioral_coordinates_kernel(
    const float* ca_final_state,        // [GRID_SIZE × GRID_SIZE × CHANNELS]
    const float* stigmergic_trace,      // [GRID_SIZE × GRID_SIZE × NUM_STIGMERGY_LAYERS]
    const float* flow_magnitude_trace,  // [GRID_SIZE × GRID_SIZE]
    float* behavioral_coords,           // [BEHAVIOR_DIM]
    curandState* rand_states,
    int organism_id
) {
    using GridStride = GridStride1D;

    // Reservoir sampling with differential weighting
    __shared__ float reservoir[BEHAVIOR_DIM];
    __shared__ float weights[BEHAVIOR_DIM];

    int tid = threadIdx.x;
    if (tid < BEHAVIOR_DIM) {
        reservoir[tid] = 0.0f;
        weights[tid] = 0.0f;
    }
    __syncthreads();

    curandState local_state = rand_states[organism_id * blockDim.x + threadIdx.x];

    // Sample spatial features with gradient-based weighting
    for (int idx = GridStride::start(); idx < GRID_SIZE * GRID_SIZE; idx += GridStride::stride()) {
        int y = idx / GRID_SIZE;
        int x = idx % GRID_SIZE;
        int cell_idx = y * GRID_SIZE + x;

        // Compute local gradient magnitude (importance weight)
        float grad_mag = flow_magnitude_trace[cell_idx];
        float stigmergy_activity = 0.0f;

        #pragma unroll
        for (int layer = 0; layer < NUM_STIGMERGY_LAYERS; layer++) {
            stigmergy_activity += stigmergic_trace[cell_idx * NUM_STIGMERGY_LAYERS + layer];
        }

        float importance = grad_mag + 0.1f * stigmergy_activity;

        // Reservoir update with probability proportional to importance
        for (int b = 0; b < BEHAVIOR_DIM; b++) {
            float threshold = importance / (weights[b] + importance + 1e-8f);
            if (curand_uniform(&local_state) < threshold) {
                int channel = (b * CHANNELS) / BEHAVIOR_DIM;
                atomicExch(&reservoir[b], ca_final_state[cell_idx * CHANNELS + channel]);
                atomicAdd(&weights[b], importance);
            }
        }
    }

    __syncthreads();

    // Write normalized coordinates
    if (tid < BEHAVIOR_DIM) {
        behavioral_coords[tid] = reservoir[tid];
    }

    rand_states[organism_id * blockDim.x + threadIdx.x] = local_state;
}

// Compute nearest Voronoi cell in behavioral space
__device__ int find_nearest_cell(
    const float* behavior,
    const VoronoiCell* cells,
    int num_cells
) {
    float min_dist_sq = FLT_MAX;
    int nearest_idx = 0;

    for (int i = 0; i < num_cells; i++) {
        float dist_sq = behavioral_distance_sq(behavior, cells[i].centroid);

        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            nearest_idx = i;
        }
    }

    return nearest_idx;
}

// Compress genome using SVD with adaptive rank selection
__global__ void compress_genome_kernel(
    const Genome* source_genome,
    CompressedGenome* compressed,
    float compression_threshold
) {
    // Construct weight matrix: [NUM_SEGMENTS × SEGMENT_PAYLOAD_SIZE]
    __shared__ float weight_matrix[NUM_SEGMENTS * SEGMENT_PAYLOAD_SIZE];

    using GridStride = GridStride1D;
    for (int idx = GridStride::start(); idx < NUM_SEGMENTS * SEGMENT_PAYLOAD_SIZE; idx += GridStride::stride()) {
        int seg = idx / SEGMENT_PAYLOAD_SIZE;
        int pos = idx % SEGMENT_PAYLOAD_SIZE;
        weight_matrix[idx] = source_genome->segments[seg].payload[pos];
    }
    __syncthreads();

    // Allocate SVD workspace
    extern __shared__ float svd_workspace[];
    float* U = svd_workspace;
    float* S = U + NUM_SEGMENTS * NUM_SEGMENTS;
    float* Vt = S + NUM_SEGMENTS;

    // Compute SVD using Jacobi method
    if (threadIdx.x == 0) {
        svd_jacobi_device(
            weight_matrix,
            U,
            S,
            Vt,
            NUM_SEGMENTS,
            SEGMENT_PAYLOAD_SIZE,
            1e-6f,
            100
        );

        // Determine effective rank
        float total_energy = 0.0f;
        for (int i = 0; i < NUM_SEGMENTS; i++) {
            total_energy += S[i] * S[i];
        }

        float cumulative_energy = 0.0f;
        int rank = 0;
        for (int i = 0; i < NUM_SEGMENTS; i++) {
            cumulative_energy += S[i] * S[i];
            if (cumulative_energy / total_energy >= compression_threshold) {
                rank = i + 1;
                break;
            }
        }
        compressed->rank = rank;

        // Store compressed representation
        for (int i = 0; i < NUM_SEGMENTS * rank; i++) {
            compressed->U_matrix[i] = U[i];
        }
        for (int i = 0; i < rank; i++) {
            compressed->singular_values[i] = S[i];
        }
        for (int i = 0; i < rank * SEGMENT_PAYLOAD_SIZE; i++) {
            compressed->Vt_matrix[i] = Vt[i];
        }

        // Compute hash and permission fingerprint
        compressed->hash = compute_genome_hash(source_genome);
        for (int i = 0; i < NUM_SEGMENTS; i++) {
            compressed->permission_fingerprint[i] = source_genome->segments[i].permission_level;
        }
    }
}

// Decompress genome from SVD representation
__device__ void decompress_genome(
    const CompressedGenome* compressed,
    Genome* target_genome
) {
    // Reconstruct weight matrix: W ≈ U · diag(S) · V^T
    for (int seg = 0; seg < NUM_SEGMENTS; seg++) {
        for (int pos = 0; pos < SEGMENT_PAYLOAD_SIZE; pos++) {
            float value = 0.0f;
            for (int r = 0; r < compressed->rank; r++) {
                value += compressed->U_matrix[seg * compressed->rank + r]
                       * compressed->singular_values[r]
                       * compressed->Vt_matrix[r * SEGMENT_PAYLOAD_SIZE + pos];
            }
            target_genome->segments[seg].payload[pos] = value;
        }

        // Restore metadata (not compressed)
        target_genome->segments[seg].permission_level = compressed->permission_fingerprint[seg];
    }
}

// Insert organism into archive if it qualifies
__global__ void archive_insertion_kernel(
    Archive* archive,
    const Genome* candidate_genome,
    float candidate_fitness,
    const float* candidate_behavior,
    uint32_t generation,
    bool* insertion_success
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Find nearest Voronoi cell
    int cell_idx = find_nearest_cell(candidate_behavior, archive->cells, archive->num_cells);
    VoronoiCell* cell = &archive->cells[cell_idx];

    bool should_insert = false;

    if (!cell->occupied) {
        should_insert = true;
    } else {
        // Check hash for deduplication
        uint64_t candidate_hash = compute_genome_hash(candidate_genome);
        if (candidate_hash == cell->elite_genome.hash) {
            *insertion_success = false;
            return;  // Duplicate genome
        }

        // Replace if better fitness
        if (candidate_fitness > cell->elite_fitness) {
            should_insert = true;
        }
    }

    if (should_insert) {
        // Compress and store genome
        compress_genome_kernel<<<1, NUM_SEGMENTS * SEGMENT_PAYLOAD_SIZE,
            (NUM_SEGMENTS * NUM_SEGMENTS + NUM_SEGMENTS + NUM_SEGMENTS * SEGMENT_PAYLOAD_SIZE) * sizeof(float)>>>(
            candidate_genome,
            &cell->elite_genome,
            0.95f  // Preserve 95% of variance
        );
        cudaDeviceSynchronize();

        cell->elite_fitness = candidate_fitness;
        for (int d = 0; d < BEHAVIOR_DIM; d++) {
            cell->elite_behavior[d] = candidate_behavior[d];
        }
        cell->occupation_count++;
        cell->last_update_generation = generation;
        cell->occupied = true;

        atomicAdd(&archive->successful_insertions, 1);
        *insertion_success = true;
    } else {
        *insertion_success = false;
    }

    atomicAdd(&archive->total_insertions, 1);
}

// Sample elite from archive with fitness-weighted selection
__device__ void sample_elite(
    const Archive* archive,
    Genome* target_genome,
    curandState* rand_state
) {
    // Count occupied cells
    int occupied_count = 0;
    for (int i = 0; i < archive->num_cells; i++) {
        if (archive->cells[i].occupied) {
            occupied_count++;
        }
    }

    if (occupied_count == 0) return;

    // Fitness-weighted sampling
    float total_fitness = 0.0f;
    for (int i = 0; i < archive->num_cells; i++) {
        if (archive->cells[i].occupied) {
            total_fitness += fmaxf(0.0f, archive->cells[i].elite_fitness);
        }
    }

    float sample = curand_uniform(rand_state) * total_fitness;
    float cumulative = 0.0f;

    for (int i = 0; i < archive->num_cells; i++) {
        if (archive->cells[i].occupied) {
            cumulative += fmaxf(0.0f, archive->cells[i].elite_fitness);
            if (cumulative >= sample) {
                decompress_genome(&archive->cells[i].elite_genome, target_genome);
                return;
            }
        }
    }
}

// Initialize archive with CVT (Centroidal Voronoi Tessellation)
__global__ void initialize_archive_kernel(
    Archive* archive,
    int num_cells,
    curandState* rand_states,
    int num_iterations
) {
    using GridStride = GridStride1D;

    curandState local_state = rand_states[blockIdx.x * blockDim.x + threadIdx.x];

    // Random initialization of centroids
    for (int i = GridStride::start(); i < num_cells; i += GridStride::stride()) {
        for (int d = 0; d < BEHAVIOR_DIM; d++) {
            archive->cells[i].centroid[d] = curand_uniform(&local_state) *
                (archive->behavioral_bounds_max[d] - archive->behavioral_bounds_min[d]) +
                archive->behavioral_bounds_min[d];
        }
        archive->cells[i].occupied = false;
        archive->cells[i].occupation_count = 0;
        archive->cells[i].elite_fitness = -FLT_MAX;
    }
    __syncthreads();

    // Lloyd's algorithm for CVT
    for (int iter = 0; iter < num_iterations; iter++) {
        __shared__ float new_centroids[BEHAVIOR_DIM * 64];  // Assume max 64 cells per block
        __shared__ int counts[64];

        int local_idx = threadIdx.x;
        if (local_idx < 64) {
            counts[local_idx] = 0;
            for (int d = 0; d < BEHAVIOR_DIM; d++) {
                new_centroids[local_idx * BEHAVIOR_DIM + d] = 0.0f;
            }
        }
        __syncthreads();

        // Sample points and assign to nearest centroid
        constexpr int NUM_SAMPLES = 1000;
        for (int s = 0; s < NUM_SAMPLES; s++) {
            float point[BEHAVIOR_DIM];
            for (int d = 0; d < BEHAVIOR_DIM; d++) {
                point[d] = curand_uniform(&local_state) *
                    (archive->behavioral_bounds_max[d] - archive->behavioral_bounds_min[d]) +
                    archive->behavioral_bounds_min[d];
            }

            int nearest = find_nearest_cell(point, archive->cells, num_cells);
            int block_local = nearest % 64;

            atomicAdd(&counts[block_local], 1);
            for (int d = 0; d < BEHAVIOR_DIM; d++) {
                atomicAdd(&new_centroids[block_local * BEHAVIOR_DIM + d], point[d]);
            }
        }
        __syncthreads();

        // Update centroids
        for (int i = GridStride::start(); i < num_cells; i += GridStride::stride()) {
            int block_local = i % 64;
            if (counts[block_local] > 0) {
                for (int d = 0; d < BEHAVIOR_DIM; d++) {
                    archive->cells[i].centroid[d] = new_centroids[block_local * BEHAVIOR_DIM + d] / counts[block_local];
                }
            }
        }
        __syncthreads();
    }

    rand_states[blockIdx.x * blockDim.x + threadIdx.x] = local_state;
}

// Compute archive saturation (percentage of occupied cells)
__device__ float compute_archive_saturation(const Archive* archive) {
    int occupied = 0;
    for (int i = 0; i < archive->num_cells; i++) {
        if (archive->cells[i].occupied) {
            occupied++;
        }
    }
    return (float)occupied / (float)archive->num_cells;
}

// Compute archive quality dispersion (variance in elite fitness)
__device__ float compute_archive_dispersion(const Archive* archive) {
    float sum = 0.0f;
    float sum_sq = 0.0f;
    int count = 0;

    for (int i = 0; i < archive->num_cells; i++) {
        if (archive->cells[i].occupied) {
            float f = archive->cells[i].elite_fitness;
            sum += f;
            sum_sq += f * f;
            count++;
        }
    }

    if (count < 2) return 0.0f;

    float mean = sum / count;
    float variance = (sum_sq / count) - (mean * mean);
    return sqrtf(fmaxf(0.0f, variance));
}
