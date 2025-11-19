#include "../config/constants.cuh"
#include "../core/types.cuh"
#include "../memory/archive.cu"
#include "../utils/tile_ops.cuh"
#include "../kernels/utils.cu"
#include <cuda_runtime.h>

// Population-level metrics for quality diversity and evolutionary dynamics

struct PopulationMetrics {
    float archive_saturation;        // Percentage of occupied Voronoi cells
    float behavioral_diversity;      // Mean distance to k-nearest neighbors in behavior space
    float fitness_variance;          // Variance of elite fitness values
    float novelty_score;            // Average novelty of recent additions
    float genome_diversity;         // Average pairwise genome distance
    float effective_population_size; // N_e based on fitness distribution
    float selection_pressure;       // Fitness range / mean fitness
    float stagnation_score;         // Timesteps since last archive improvement
};

// Compute behavioral diversity using k-nearest neighbors
__global__ void compute_behavioral_diversity_kernel(
    const Archive* archive,
    float* diversity_score,
    int k_neighbors
) {
    using GridStride = GridStride1D;

    __shared__ float distances[MAX_ARCHIVE_CELLS];
    __shared__ float total_diversity;

    if (threadIdx.x == 0) {
        total_diversity = 0.0f;
    }
    __syncthreads();

    // For each occupied cell, find k-nearest neighbors
    int occupied_count = 0;
    for (int i = 0; i < archive->num_cells; i++) {
        if (archive->cells[i].occupied) {
            occupied_count++;
        }
    }

    if (occupied_count < k_neighbors) {
        if (threadIdx.x == 0) {
            *diversity_score = 0.0f;
        }
        return;
    }

    for (int i = GridStride::start(); i < archive->num_cells; i += GridStride::stride()) {
        if (!archive->cells[i].occupied) continue;

        const float* behavior_i = archive->cells[i].elite_behavior;

        // Compute distances to all other occupied cells using behavioral_distance_sq from utils
        int dist_count = 0;
        for (int j = 0; j < archive->num_cells; j++) {
            if (i == j || !archive->cells[j].occupied) continue;

            float dist_sq = behavioral_distance_sq(behavior_i, archive->cells[j].elite_behavior);
            distances[dist_count++] = sqrtf(dist_sq);
        }

        // Sort distances to find k-nearest (bubble sort for small k)
        for (int a = 0; a < k_neighbors; a++) {
            for (int b = a + 1; b < dist_count; b++) {
                if (distances[b] < distances[a]) {
                    float temp = distances[a];
                    distances[a] = distances[b];
                    distances[b] = temp;
                }
            }
        }

        // Average distance to k-nearest
        float avg_k_dist = 0.0f;
        for (int n = 0; n < k_neighbors; n++) {
            avg_k_dist += distances[n];
        }
        avg_k_dist /= k_neighbors;

        atomicAdd(&total_diversity, avg_k_dist);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        *diversity_score = total_diversity / fmaxf(1.0f, (float)occupied_count);
    }
}

// Compute genome diversity (average pairwise Hamming distance)
__global__ void compute_genome_diversity_kernel(
    const Genome* population,
    int population_size,
    float* diversity_score
) {
    using GridStride = GridStride1D;
    using BlockReduce = BlockReduce<float>;

    __shared__ float sum_distances;
    if (threadIdx.x == 0) {
        sum_distances = 0.0f;
    }
    __syncthreads();

    float local_sum = 0.0f;
    int pair_count = 0;

    // Sample pairs to compute average pairwise distance
    for (int idx = GridStride::start(); idx < population_size * (population_size - 1) / 2; idx += GridStride::stride()) {
        // Map linear index to (i, j) pair where i < j
        int i = 0;
        int remaining = idx;
        while (remaining >= (population_size - 1 - i)) {
            remaining -= (population_size - 1 - i);
            i++;
        }
        int j = i + 1 + remaining;

        // Compute genome distance
        float distance = 0.0f;
        for (int seg = 0; seg < NUM_SEGMENTS; seg++) {
            for (int w = 0; w < SEGMENT_PAYLOAD_SIZE; w++) {
                float delta = population[i].segments[seg].payload[w] -
                             population[j].segments[seg].payload[w];
                distance += delta * delta;
            }
        }
        distance = sqrtf(distance);

        local_sum += distance;
        pair_count++;
    }

    float reduced_sum = BlockReduce::sum(local_sum);
    if (threadIdx.x == 0) {
        atomicAdd(&sum_distances, reduced_sum);
    }
    __syncthreads();

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int total_pairs = population_size * (population_size - 1) / 2;
        *diversity_score = sum_distances / fmaxf(1.0f, (float)total_pairs);
    }
}

// Compute effective population size N_e using fitness variance
__global__ void compute_effective_population_size_kernel(
    const float* fitness_values,
    int population_size,
    float* effective_size
) {
    using BlockReduce = BlockReduce<float>;

    __shared__ float sum_fitness;
    __shared__ float sum_fitness_sq;

    if (threadIdx.x == 0) {
        sum_fitness = 0.0f;
        sum_fitness_sq = 0.0f;
    }
    __syncthreads();

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    for (int i = threadIdx.x; i < population_size; i += blockDim.x) {
        float f = fitness_values[i];
        local_sum += f;
        local_sum_sq += f * f;
    }

    float reduced_sum = BlockReduce::sum(local_sum);
    float reduced_sum_sq = BlockReduce::sum(local_sum_sq);

    if (threadIdx.x == 0) {
        atomicAdd(&sum_fitness, reduced_sum);
        atomicAdd(&sum_fitness_sq, reduced_sum_sq);
    }
    __syncthreads();

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float mean = sum_fitness / population_size;
        float variance = (sum_fitness_sq / population_size) - (mean * mean);

        // N_e approximation: N_e ≈ (mean^2) / variance
        if (variance > 1e-6f) {
            *effective_size = (mean * mean) / variance;
        } else {
            *effective_size = (float)population_size;
        }
    }
}

// Compute novelty score for recent archive additions
__global__ void compute_novelty_score_kernel(
    const Archive* archive,
    uint32_t current_generation,
    int recency_window,
    float* novelty_score
) {
    using GridStride = GridStride1D;

    __shared__ float sum_novelty;
    if (threadIdx.x == 0) {
        sum_novelty = 0.0f;
    }
    __syncthreads();

    int recent_count = 0;
    float local_novelty = 0.0f;

    // For each recent addition, compute average distance to archive
    for (int i = GridStride::start(); i < archive->num_cells; i += GridStride::stride()) {
        if (!archive->cells[i].occupied) continue;

        uint32_t age = current_generation - archive->cells[i].last_update_generation;
        if (age > recency_window) continue;

        const float* behavior_i = archive->cells[i].elite_behavior;
        float avg_dist = 0.0f;
        int other_count = 0;

        for (int j = 0; j < archive->num_cells; j++) {
            if (i == j || !archive->cells[j].occupied) continue;

            float dist_sq = behavioral_distance_sq(behavior_i, archive->cells[j].elite_behavior);
            avg_dist += sqrtf(dist_sq);
            other_count++;
        }

        if (other_count > 0) {
            local_novelty += avg_dist / other_count;
            recent_count++;
        }
    }

    using WarpReduce = WarpReduce<float>;
    local_novelty = WarpReduce::sum(local_novelty);

    if (threadIdx.x % WARP_SIZE == 0) {
        atomicAdd(&sum_novelty, local_novelty);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        *novelty_score = sum_novelty / fmaxf(1.0f, (float)recent_count);
    }
}

// Compute selection pressure metric
__global__ void compute_selection_pressure_kernel(
    const Archive* archive,
    float* selection_pressure
) {
    __shared__ float min_fitness;
    __shared__ float max_fitness;
    __shared__ float sum_fitness;
    __shared__ int count;

    if (threadIdx.x == 0) {
        min_fitness = FLT_MAX;
        max_fitness = -FLT_MAX;
        sum_fitness = 0.0f;
        count = 0;
    }
    __syncthreads();

    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;
    int local_count = 0;

    for (int i = threadIdx.x; i < archive->num_cells; i += blockDim.x) {
        if (archive->cells[i].occupied) {
            float f = archive->cells[i].elite_fitness;
            local_min = fminf(local_min, f);
            local_max = fmaxf(local_max, f);
            local_sum += f;
            local_count++;
        }
    }

    using WarpReduce = WarpReduce<float>;
    local_min = WarpReduce::min(local_min);
    local_max = WarpReduce::max(local_max);
    local_sum = WarpReduce::sum(local_sum);

    if (threadIdx.x % WARP_SIZE == 0) {
        atomicMinFloat(&min_fitness, local_min);
        atomicMaxFloat(&max_fitness, local_max);
        atomicAdd(&sum_fitness, local_sum);
        atomicAdd(&count, local_count);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float mean = sum_fitness / fmaxf(1.0f, (float)count);
        float range = max_fitness - min_fitness;
        *selection_pressure = range / fmaxf(1e-6f, fabsf(mean));
    }
}

// Detect stagnation (no improvement in top fitness)
__global__ void compute_stagnation_score_kernel(
    const Archive* archive,
    uint32_t current_generation,
    float* stagnation_score
) {
    __shared__ float best_fitness;
    __shared__ uint32_t best_generation;

    if (threadIdx.x == 0) {
        best_fitness = -FLT_MAX;
        best_generation = 0;
    }
    __syncthreads();

    float local_best_fitness = -FLT_MAX;
    uint32_t local_best_generation = 0;

    for (int i = threadIdx.x; i < archive->num_cells; i += blockDim.x) {
        if (archive->cells[i].occupied && archive->cells[i].elite_fitness > local_best_fitness) {
            local_best_fitness = archive->cells[i].elite_fitness;
            local_best_generation = archive->cells[i].last_update_generation;
        }
    }

    using WarpReduce = WarpReduce<float>;
    local_best_fitness = WarpReduce::max(local_best_fitness);

    if (threadIdx.x % WARP_SIZE == 0) {
        atomicMaxFloat(&best_fitness, local_best_fitness);
        atomicMax(&best_generation, local_best_generation);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        *stagnation_score = (float)(current_generation - best_generation);
    }
}

// Aggregate all population metrics
__global__ void compute_population_metrics_kernel(
    const Archive* archive,
    const Genome* population,
    const float* fitness_values,
    int population_size,
    uint32_t current_generation,
    PopulationMetrics* metrics
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Archive saturation
    metrics->archive_saturation = compute_archive_saturation(archive);

    // Fitness variance
    float sum_fitness = 0.0f;
    float sum_fitness_sq = 0.0f;
    int occupied_count = 0;

    for (int i = 0; i < archive->num_cells; i++) {
        if (archive->cells[i].occupied) {
            float f = archive->cells[i].elite_fitness;
            sum_fitness += f;
            sum_fitness_sq += f * f;
            occupied_count++;
        }
    }

    if (occupied_count > 0) {
        float mean = sum_fitness / occupied_count;
        metrics->fitness_variance = (sum_fitness_sq / occupied_count) - (mean * mean);
    } else {
        metrics->fitness_variance = 0.0f;
    }

    // Launch child kernels for complex metrics
    float* diversity_ptr;
    cudaMalloc(&diversity_ptr, sizeof(float));

    compute_behavioral_diversity_kernel<<<1, 256>>>(archive, diversity_ptr, 5);
    cudaDeviceSynchronize();
    cudaMemcpy(&metrics->behavioral_diversity, diversity_ptr, sizeof(float), cudaMemcpyDeviceToDevice);

    compute_genome_diversity_kernel<<<32, 256>>>(population, population_size, diversity_ptr);
    cudaDeviceSynchronize();
    cudaMemcpy(&metrics->genome_diversity, diversity_ptr, sizeof(float), cudaMemcpyDeviceToDevice);

    compute_effective_population_size_kernel<<<1, 256>>>(fitness_values, population_size, diversity_ptr);
    cudaDeviceSynchronize();
    cudaMemcpy(&metrics->effective_population_size, diversity_ptr, sizeof(float), cudaMemcpyDeviceToDevice);

    compute_novelty_score_kernel<<<1, 256>>>(archive, current_generation, 100, diversity_ptr);
    cudaDeviceSynchronize();
    cudaMemcpy(&metrics->novelty_score, diversity_ptr, sizeof(float), cudaMemcpyDeviceToDevice);

    compute_selection_pressure_kernel<<<1, 256>>>(archive, diversity_ptr);
    cudaDeviceSynchronize();
    cudaMemcpy(&metrics->selection_pressure, diversity_ptr, sizeof(float), cudaMemcpyDeviceToDevice);

    compute_stagnation_score_kernel<<<1, 256>>>(archive, current_generation, diversity_ptr);
    cudaDeviceSynchronize();
    cudaMemcpy(&metrics->stagnation_score, diversity_ptr, sizeof(float), cudaMemcpyDeviceToDevice);

    cudaFree(diversity_ptr);
}

// Compute archive coverage (percentage of behavioral space explored)
__global__ void compute_behavioral_coverage_kernel(
    const Archive* archive,
    float* coverage_score
) {
    // Estimate coverage as volume of convex hull / total volume
    __shared__ float min_coords[BEHAVIOR_DIM];
    __shared__ float max_coords[BEHAVIOR_DIM];

    if (threadIdx.x < BEHAVIOR_DIM) {
        min_coords[threadIdx.x] = FLT_MAX;
        max_coords[threadIdx.x] = -FLT_MAX;
    }
    __syncthreads();

    // Find bounding box of occupied cells
    for (int i = threadIdx.x; i < archive->num_cells; i += blockDim.x) {
        if (!archive->cells[i].occupied) continue;

        for (int d = 0; d < BEHAVIOR_DIM; d++) {
            atomicMinFloat(&min_coords[d], archive->cells[i].elite_behavior[d]);
            atomicMaxFloat(&max_coords[d], archive->cells[i].elite_behavior[d]);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // Compute occupied volume / total volume
        float occupied_volume = 1.0f;
        float total_volume = 1.0f;

        for (int d = 0; d < BEHAVIOR_DIM; d++) {
            occupied_volume *= (max_coords[d] - min_coords[d]);
            total_volume *= (archive->behavioral_bounds_max[d] - archive->behavioral_bounds_min[d]);
        }

        *coverage_score = occupied_volume / fmaxf(1e-10f, total_volume);
    }
}

// Compute lineage depth distribution
__global__ void compute_lineage_depth_kernel(
    const Genome* population,
    int population_size,
    float* avg_lineage_depth,
    float* max_lineage_depth
) {
    using BlockReduce = BlockReduce<float>;

    __shared__ float sum_depth;
    __shared__ float max_depth_shared;

    if (threadIdx.x == 0) {
        sum_depth = 0.0f;
        max_depth_shared = 0.0f;
    }
    __syncthreads();

    float local_sum = 0.0f;
    float local_max = 0.0f;

    for (int i = threadIdx.x; i < population_size; i += blockDim.x) {
        float depth = population[i].lineage_depth;
        local_sum += depth;
        local_max = fmaxf(local_max, depth);
    }

    float reduced_sum = BlockReduce::sum(local_sum);
    float reduced_max = BlockReduce::max(local_max);

    if (threadIdx.x == 0) {
        atomicAdd(&sum_depth, reduced_sum);
        atomicMaxFloat(&max_depth_shared, reduced_max);
    }
    __syncthreads();

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *avg_lineage_depth = sum_depth / population_size;
        *max_lineage_depth = max_depth_shared;
    }
}
