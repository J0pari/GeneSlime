#include "../config/constants.cuh"
#include "../core/types.cuh"
#include "../memory/archive.cu"
#include "../kernels/utils.cu"
#include "../utils/tile_ops.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Parent selection and elite sampling strategies

// Tournament selection with archive-biased sampling
__global__ void select_parents_tournament_kernel(
    const Organism* population,
    const Archive* archive,
    Genome* parent_a,
    Genome* parent_b,
    int population_size,
    int tournament_size,
    float archive_sample_rate,
    curandState* rand_states,
    int organism_id
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    curandState local_state = rand_states[organism_id];

    // Decide whether to sample from archive or population
    bool use_archive_a = (curand_uniform(&local_state) < archive_sample_rate);
    bool use_archive_b = (curand_uniform(&local_state) < archive_sample_rate);

    // Select parent A
    if (use_archive_a && archive->successful_insertions > 0) {
        sample_elite(archive, parent_a, &local_state);
    } else {
        // Tournament selection from population
        int best_idx = curand(&local_state) % population_size;
        float best_fitness = population[best_idx].fitness;

        for (int t = 1; t < tournament_size; t++) {
            int candidate_idx = curand(&local_state) % population_size;
            float candidate_fitness = population[candidate_idx].fitness;

            if (candidate_fitness > best_fitness) {
                best_idx = candidate_idx;
                best_fitness = candidate_fitness;
            }
        }

        *parent_a = population[best_idx].genome;
    }

    // Select parent B
    if (use_archive_b && archive->successful_insertions > 0) {
        sample_elite(archive, parent_b, &local_state);
    } else {
        int best_idx = curand(&local_state) % population_size;
        float best_fitness = population[best_idx].fitness;

        for (int t = 1; t < tournament_size; t++) {
            int candidate_idx = curand(&local_state) % population_size;
            float candidate_fitness = population[candidate_idx].fitness;

            if (candidate_fitness > best_fitness) {
                best_idx = candidate_idx;
                best_fitness = candidate_fitness;
            }
        }

        *parent_b = population[best_idx].genome;
    }

    rand_states[organism_id] = local_state;
}

// Fitness-proportional selection (roulette wheel)
__global__ void select_parents_roulette_kernel(
    const Organism* population,
    Genome* parent_a,
    Genome* parent_b,
    int population_size,
    curandState* rand_states,
    int organism_id
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Compute total fitness
    float total_fitness = 0.0f;
    float min_fitness = FLT_MAX;

    for (int i = 0; i < population_size; i++) {
        float f = population[i].fitness;
        min_fitness = fminf(min_fitness, f);
    }

    // Shift fitness to ensure all positive
    float shift = (min_fitness < 0.0f) ? -min_fitness + 1e-6f : 0.0f;

    for (int i = 0; i < population_size; i++) {
        total_fitness += population[i].fitness + shift;
    }

    curandState local_state = rand_states[organism_id];

    // Sample parent A
    float sample_a = curand_uniform(&local_state) * total_fitness;
    float cumulative = 0.0f;
    for (int i = 0; i < population_size; i++) {
        cumulative += population[i].fitness + shift;
        if (cumulative >= sample_a) {
            *parent_a = population[i].genome;
            break;
        }
    }

    // Sample parent B
    float sample_b = curand_uniform(&local_state) * total_fitness;
    cumulative = 0.0f;
    for (int i = 0; i < population_size; i++) {
        cumulative += population[i].fitness + shift;
        if (cumulative >= sample_b) {
            *parent_b = population[i].genome;
            break;
        }
    }

    rand_states[organism_id] = local_state;
}

// Novelty-based selection using behavioral distance
__global__ void select_parents_novelty_kernel(
    const Organism* population,
    Genome* parent_a,
    Genome* parent_b,
    int population_size,
    int k_nearest,
    curandState* rand_states,
    int organism_id
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    extern __shared__ float novelty_scores[];

    // Compute novelty for each organism (average distance to k-nearest neighbors)
    for (int i = 0; i < population_size; i++) {
        float distances[MAX_POPULATION_SIZE];
        int dist_count = 0;

        for (int j = 0; j < population_size; j++) {
            if (i == j) continue;

            float dist_sq = behavioral_distance_sq(
                population[i].behavioral_coords,
                population[j].behavioral_coords
            );
            distances[dist_count++] = sqrtf(dist_sq);
        }

        // Sort to find k-nearest
        for (int a = 0; a < k_nearest && a < dist_count; a++) {
            for (int b = a + 1; b < dist_count; b++) {
                if (distances[b] < distances[a]) {
                    float temp = distances[a];
                    distances[a] = distances[b];
                    distances[b] = temp;
                }
            }
        }

        // Average distance to k-nearest
        float novelty = 0.0f;
        int limit = (k_nearest < dist_count) ? k_nearest : dist_count;
        for (int n = 0; n < limit; n++) {
            novelty += distances[n];
        }
        novelty_scores[i] = (limit > 0) ? (novelty / limit) : 0.0f;
    }

    // Novelty-proportional sampling
    float total_novelty = 0.0f;
    for (int i = 0; i < population_size; i++) {
        total_novelty += novelty_scores[i];
    }

    curandState local_state = rand_states[organism_id];

    // Sample parent A
    float sample_a = curand_uniform(&local_state) * total_novelty;
    float cumulative = 0.0f;
    for (int i = 0; i < population_size; i++) {
        cumulative += novelty_scores[i];
        if (cumulative >= sample_a) {
            *parent_a = population[i].genome;
            break;
        }
    }

    // Sample parent B
    float sample_b = curand_uniform(&local_state) * total_novelty;
    cumulative = 0.0f;
    for (int i = 0; i < population_size; i++) {
        cumulative += novelty_scores[i];
        if (cumulative >= sample_b) {
            *parent_b = population[i].genome;
            break;
        }
    }

    rand_states[organism_id] = local_state;
}

// Elitism: copy top N organisms directly to next generation
__global__ void elitism_copy_kernel(
    const Organism* population,
    Organism* next_generation,
    int population_size,
    int num_elites
) {
    using GridStride = GridStride1D;

    // Find top fitness organisms
    extern __shared__ int elite_indices[];

    if (threadIdx.x == 0) {
        for (int e = 0; e < num_elites; e++) {
            float best_fitness = -FLT_MAX;
            int best_idx = -1;

            for (int i = 0; i < population_size; i++) {
                bool already_selected = false;
                for (int prev = 0; prev < e; prev++) {
                    if (elite_indices[prev] == i) {
                        already_selected = true;
                        break;
                    }
                }

                if (!already_selected && population[i].fitness > best_fitness) {
                    best_fitness = population[i].fitness;
                    best_idx = i;
                }
            }

            elite_indices[e] = best_idx;
        }
    }
    __syncthreads();

    // Copy elite genomes
    for (int e = GridStride::start(); e < num_elites; e += GridStride::stride()) {
        if (elite_indices[e] >= 0) {
            next_generation[e] = population[elite_indices[e]];
        }
    }
}

// Adaptive selection pressure based on population metrics
__device__ float compute_selection_pressure_adaptive(
    float archive_saturation,
    float genome_diversity,
    float stagnation_score,
    float base_tournament_size
) {
    // Increase tournament size (selection pressure) when:
    // - Archive is saturated (exploitation phase)
    // - Diversity is high (can afford selection)
    // - Stagnation is low (still making progress)

    float pressure_multiplier = 1.0f;

    if (archive_saturation > 0.8f) {
        pressure_multiplier *= 1.5f;  // High saturation → exploit
    }

    if (genome_diversity < 0.1f) {
        pressure_multiplier *= 0.5f;  // Low diversity → reduce pressure
    }

    if (stagnation_score > 100.0f) {
        pressure_multiplier *= 0.7f;  // Stagnating → reduce pressure, explore more
    }

    return base_tournament_size * pressure_multiplier;
}

// Hybrid selection: combines multiple strategies
__global__ void select_parents_hybrid_kernel(
    const Organism* population,
    const Archive* archive,
    const PopulationMetrics* metrics,
    Genome* parent_a,
    Genome* parent_b,
    int population_size,
    curandState* rand_states,
    int organism_id
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    curandState local_state = rand_states[organism_id];

    // Adaptive strategy selection based on metrics
    float archive_bias = clamp(metrics->archive_saturation, 0.1f, 0.5f);
    float novelty_weight = (metrics->genome_diversity < 0.2f) ? 0.6f : 0.2f;
    float fitness_weight = 1.0f - novelty_weight;

    float strategy_roll = curand_uniform(&local_state);

    if (strategy_roll < archive_bias) {
        // Sample from archive
        sample_elite(archive, parent_a, &local_state);
        sample_elite(archive, parent_b, &local_state);
    } else {
        // Combined fitness + novelty selection
        float scores[MAX_POPULATION_SIZE];

        for (int i = 0; i < population_size; i++) {
            // Compute novelty (simplified k=5)
            float novelty = 0.0f;
            int k = 5;
            float distances[MAX_POPULATION_SIZE];

            for (int j = 0; j < population_size; j++) {
                if (i == j) continue;
                float dist_sq = behavioral_distance_sq(
                    population[i].behavioral_coords,
                    population[j].behavioral_coords
                );
                distances[j] = sqrtf(dist_sq);
            }

            // Partial sort for k-nearest
            for (int n = 0; n < k && n < population_size - 1; n++) {
                novelty += distances[n];
            }
            novelty /= fminf((float)k, (float)(population_size - 1));

            // Normalize fitness
            float fitness_norm = (population[i].fitness + 10.0f) / 20.0f;

            scores[i] = fitness_weight * fitness_norm + novelty_weight * novelty;
        }

        // Sample proportionally to combined score
        float total_score = 0.0f;
        for (int i = 0; i < population_size; i++) {
            total_score += fmaxf(0.0f, scores[i]);
        }

        float sample_a = curand_uniform(&local_state) * total_score;
        float cumulative = 0.0f;
        for (int i = 0; i < population_size; i++) {
            cumulative += fmaxf(0.0f, scores[i]);
            if (cumulative >= sample_a) {
                *parent_a = population[i].genome;
                break;
            }
        }

        float sample_b = curand_uniform(&local_state) * total_score;
        cumulative = 0.0f;
        for (int i = 0; i < population_size; i++) {
            cumulative += fmaxf(0.0f, scores[i]);
            if (cumulative >= sample_b) {
                *parent_b = population[i].genome;
                break;
            }
        }
    }

    rand_states[organism_id] = local_state;
}
