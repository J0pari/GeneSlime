#include "../config/constants.cuh"
#include "../core/types.cuh"
#include "../metrics/population.cu"
#include "../kernels/utils.cu"
#include "../utils/tile_ops.cuh"
#include <cuda_runtime.h>

// Regulatory mechanisms: crisis detection and adaptive parameter modulation

struct RegulatoryState {
    bool lifecycle_crisis;
    float mutation_rate_multiplier;
    float structural_mutation_rate;
    float recombination_rate;
    float archive_sample_rate;
    int crisis_cooldown;
};

// Detect lifecycle crisis conditions
__global__ void detect_crisis_kernel(
    const PopulationMetrics* metrics,
    const Archive* archive,
    RegulatoryState* regulatory,
    uint32_t generation
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    bool crisis_detected = false;

    // Crisis condition 1: Stagnation (no improvement in 200 generations)
    if (metrics->stagnation_score > 200.0f) {
        crisis_detected = true;
    }

    // Crisis condition 2: Extreme loss of diversity
    if (metrics->genome_diversity < 0.05f && metrics->effective_population_size < 10.0f) {
        crisis_detected = true;
    }

    // Crisis condition 3: Archive saturation plateau with low novelty
    if (metrics->archive_saturation > 0.95f && metrics->novelty_score < 0.01f) {
        crisis_detected = true;
    }

    // Crisis condition 4: Selection pressure collapse (all similar fitness)
    if (metrics->selection_pressure < 0.01f) {
        crisis_detected = true;
    }

    // Apply cooldown (don't trigger crisis too frequently)
    if (crisis_detected && regulatory->crisis_cooldown == 0) {
        regulatory->lifecycle_crisis = true;
        regulatory->crisis_cooldown = 50;  // 50 generation cooldown
    } else {
        regulatory->lifecycle_crisis = false;
        if (regulatory->crisis_cooldown > 0) {
            regulatory->crisis_cooldown--;
        }
    }
}

// Modulate mutation rates based on population state
__global__ void modulate_mutation_rates_kernel(
    const PopulationMetrics* metrics,
    RegulatoryState* regulatory
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Base rates
    float base_mutation_rate = 1.0f;
    float base_structural_rate = 0.01f;

    // Increase mutation during crisis
    if (regulatory->lifecycle_crisis) {
        regulatory->mutation_rate_multiplier = 10.0f;  // 10x mutation boost
        regulatory->structural_mutation_rate = 0.1f;   // Increase structural changes
    } else {
        // Adaptive modulation based on diversity
        if (metrics->genome_diversity < 0.1f) {
            // Low diversity → increase mutation
            regulatory->mutation_rate_multiplier = 3.0f;
            regulatory->structural_mutation_rate = 0.05f;
        } else if (metrics->genome_diversity > 0.5f) {
            // High diversity → reduce mutation (conserve good solutions)
            regulatory->mutation_rate_multiplier = 0.5f;
            regulatory->structural_mutation_rate = 0.005f;
        } else {
            // Normal diversity → baseline
            regulatory->mutation_rate_multiplier = base_mutation_rate;
            regulatory->structural_mutation_rate = base_structural_rate;
        }
    }

    // Clamp to safe ranges
    regulatory->mutation_rate_multiplier = clamp(regulatory->mutation_rate_multiplier, 0.1f, 20.0f);
    regulatory->structural_mutation_rate = clamp(regulatory->structural_mutation_rate, 0.001f, 0.2f);
}

// Modulate recombination rate
__global__ void modulate_recombination_rate_kernel(
    const PopulationMetrics* metrics,
    RegulatoryState* regulatory
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Increase recombination when diversity is high (mix good solutions)
    // Decrease when diversity is low (avoid breaking rare variants)

    if (metrics->genome_diversity > 0.4f && metrics->archive_saturation > 0.5f) {
        regulatory->recombination_rate = 0.8f;  // High mixing
    } else if (metrics->genome_diversity < 0.1f) {
        regulatory->recombination_rate = 0.3f;  // Low mixing, preserve variants
    } else {
        regulatory->recombination_rate = 0.6f;  // Baseline
    }

    // During crisis, favor recombination over mutation
    if (regulatory->lifecycle_crisis) {
        regulatory->recombination_rate = 0.9f;
    }
}

// Modulate archive sampling rate
__global__ void modulate_archive_sampling_kernel(
    const PopulationMetrics* metrics,
    RegulatoryState* regulatory
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Archive sampling rate determines how often to sample elites vs population

    if (metrics->archive_saturation < 0.2f) {
        // Archive not yet populated → sample mostly from population
        regulatory->archive_sample_rate = 0.1f;
    } else if (metrics->archive_saturation > 0.8f) {
        // Archive well-populated → leverage elites
        regulatory->archive_sample_rate = 0.5f;
    } else {
        // Moderate archive → balanced
        regulatory->archive_sample_rate = 0.3f;
    }

    // During stagnation, reduce archive sampling (explore more from population)
    if (metrics->stagnation_score > 100.0f) {
        regulatory->archive_sample_rate *= 0.5f;
    }

    regulatory->archive_sample_rate = clamp(regulatory->archive_sample_rate, 0.05f, 0.7f);
}

// Trigger mass extinction event (drastic crisis response)
__global__ void trigger_extinction_event_kernel(
    Organism* population,
    const Archive* archive,
    int population_size,
    float survival_rate,
    curandState* rand_states
) {
    using GridStride = GridStride1D;

    for (int i = GridStride::start(); i < population_size; i += GridStride::stride()) {
        curandState local_state = rand_states[i];

        if (curand_uniform(&local_state) > survival_rate) {
            // Replace with archive elite or random initialization
            if (archive->successful_insertions > 0 && curand_uniform(&local_state) < 0.7f) {
                sample_elite(archive, &population[i].genome, &local_state);
            } else {
                // Random re-initialization
                for (int seg = 0; seg < NUM_SEGMENTS; seg++) {
                    for (int w = 0; w < SEGMENT_PAYLOAD_SIZE; w++) {
                        population[i].genome.segments[seg].payload[w] =
                            curand_normal(&local_state) * 0.1f;
                    }
                }
            }

            population[i].fitness = -FLT_MAX;  // Force re-evaluation
            population[i].genome.lineage_depth = 0;
        }

        rand_states[i] = local_state;
    }
}

// Gradual parameter scheduling (annealing-like)
__global__ void anneal_parameters_kernel(
    RegulatoryState* regulatory,
    uint32_t generation,
    uint32_t max_generations
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    float progress = (float)generation / (float)max_generations;

    // Early phase (0-25%): High exploration
    if (progress < 0.25f) {
        regulatory->mutation_rate_multiplier *= 1.5f;
        regulatory->archive_sample_rate *= 0.5f;
    }
    // Mid phase (25-75%): Balanced
    else if (progress < 0.75f) {
        // Use adaptive values as-is
    }
    // Late phase (75-100%): Exploitation
    else {
        regulatory->mutation_rate_multiplier *= 0.5f;
        regulatory->archive_sample_rate *= 1.5f;
    }

    // Clamp
    regulatory->mutation_rate_multiplier = clamp(regulatory->mutation_rate_multiplier, 0.1f, 20.0f);
    regulatory->archive_sample_rate = clamp(regulatory->archive_sample_rate, 0.05f, 0.7f);
}

// Full regulatory update (orchestrates all modulation)
__global__ void update_regulatory_state_kernel(
    const PopulationMetrics* metrics,
    const Archive* archive,
    RegulatoryState* regulatory,
    uint32_t generation,
    uint32_t max_generations
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Detect crisis
    detect_crisis_kernel<<<1, 1>>>(metrics, archive, regulatory, generation);
    cudaDeviceSynchronize();

    // Modulate parameters
    modulate_mutation_rates_kernel<<<1, 1>>>(metrics, regulatory);
    modulate_recombination_rate_kernel<<<1, 1>>>(metrics, regulatory);
    modulate_archive_sampling_kernel<<<1, 1>>>(metrics, regulatory);
    cudaDeviceSynchronize();

    // Apply gradual annealing
    anneal_parameters_kernel<<<1, 1>>>(regulatory, generation, max_generations);
    cudaDeviceSynchronize();
}

// Initialize regulatory state with defaults
__global__ void init_regulatory_state_kernel(RegulatoryState* regulatory) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    regulatory->lifecycle_crisis = false;
    regulatory->mutation_rate_multiplier = 1.0f;
    regulatory->structural_mutation_rate = 0.01f;
    regulatory->recombination_rate = 0.6f;
    regulatory->archive_sample_rate = 0.3f;
    regulatory->crisis_cooldown = 0;
}

// Diversity injection: introduce random variants to prevent convergence
__global__ void inject_diversity_kernel(
    Organism* population,
    int population_size,
    float injection_rate,
    curandState* rand_states
) {
    using GridStride = GridStride1D;

    for (int i = GridStride::start(); i < population_size; i += GridStride::stride()) {
        curandState local_state = rand_states[i];

        if (curand_uniform(&local_state) < injection_rate) {
            // Inject random noise into genome
            int num_segments_to_mutate = 1 + (curand(&local_state) % 3);

            for (int s = 0; s < num_segments_to_mutate; s++) {
                int seg_idx = curand(&local_state) % NUM_SEGMENTS;

                for (int w = 0; w < SEGMENT_PAYLOAD_SIZE; w += 4) {
                    population[i].genome.segments[seg_idx].payload[w] +=
                        curand_normal(&local_state) * 0.5f;
                }
            }
        }

        rand_states[i] = local_state;
    }
}
