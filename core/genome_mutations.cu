#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "types.cuh"
#include "../utils/tile_ops.cuh"

// ============================================================================
// Permission-Based Mutation
// ============================================================================

__device__ bool can_mutate_segment(
    const GenomeSegment* segment,
    bool lifecycle_crisis,
    curandState* rand_state
) {
    float r = curand_uniform(rand_state);

    switch (segment->permission_level) {
        case LEVEL_0_PROTECTED:
            return r < MUTATION_RATE_L0;  // 0.1%

        case LEVEL_1_STRUCTURAL:
            if (!lifecycle_crisis) return false;  // Locked unless crisis
            return r < MUTATION_RATE_L1;          // 10% during crisis

        case LEVEL_2_PARAMETRIC:
            return r < MUTATION_RATE_L2;  // 100%

        case LEVEL_3_REGULATORY:
            return r < MUTATION_RATE_L3;  // 1000% (always mutates, multiple times)

        default:
            return false;
    }
}

__device__ void mutate_segment_payload(
    GenomeSegment* segment,
    curandState* rand_state
) {
    // Mutation strength depends on permission level
    float mutation_scale = 0.1f;

    if (segment->permission_level == LEVEL_3_REGULATORY) {
        mutation_scale = 0.3f; // Regulatory segments mutate more aggressively
    } else if (segment->permission_level == LEVEL_0_PROTECTED) {
        mutation_scale = 0.01f; // Protected segments mutate minimally
    }

    // Mutate payload with Gaussian noise
    for (int i = 0; i < SEGMENT_PAYLOAD_SIZE; i++) {
        if (curand_uniform(rand_state) < 0.3f) { // 30% of weights mutated
            segment->payload[i] += curand_normal(rand_state) * mutation_scale;

            // Clamp to reasonable range
            segment->payload[i] = fminf(fmaxf(segment->payload[i], -3.0f), 3.0f);
        }
    }

    // Also mutate stochastic variants
    for (int i = 0; i < SEGMENT_PAYLOAD_SIZE; i++) {
        if (curand_uniform(rand_state) < 0.2f) {
            segment->stochastic.variant_A.expansion[i] += curand_normal(rand_state) * mutation_scale * 0.5f;
            segment->stochastic.variant_B.expansion[i] += curand_normal(rand_state) * mutation_scale * 0.7f;
            segment->stochastic.variant_C.expansion[i] += curand_normal(rand_state) * mutation_scale;
        }
    }
}

__global__ void mutate_genomes_kernel(
    Genome* genomes,
    const PopulationMetrics* metrics,
    int n
) {
    int idx = GridStride::thread_id();
    if (idx < n) {
        curandState local_state;
        curand_init(CURAND_SEED, idx + metrics->generation * n, 0, &local_state);

        Genome* genome = &genomes[idx];
        bool crisis = metrics->crisis_mode;

        for (int seg_idx = 0; seg_idx < NUM_SEGMENTS; seg_idx++) {
            if (!genome->is_active(seg_idx)) continue;

            if (can_mutate_segment(&genome->segments[seg_idx], crisis, &local_state)) {
                mutate_segment_payload(&genome->segments[seg_idx], &local_state);
            }
        }

        // Recompute hash after mutation
        genome->structural_hash = compute_genome_hash(genome);
    }
}

// ============================================================================
// Recombination with Weighted Segment Sampling
// ============================================================================

__device__ void recombine_asymmetric(
    const Genome* parent_A,
    const Genome* parent_B,
    Genome* child,
    const float* fitness_A_segments,
    const float* fitness_B_segments,
    curandState* rand_state
) {
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        // Weighted probability based on segment fitness attribution
        float w_A = fitness_A_segments[i] + 1e-6f; // Add epsilon to avoid division by zero
        float w_B = fitness_B_segments[i] + 1e-6f;
        float p_A = w_A / (w_A + w_B);

        if (curand_uniform(rand_state) < p_A) {
            child->segments[i] = parent_A->segments[i];
        } else {
            child->segments[i] = parent_B->segments[i];
        }

        // Inherit activity status
        bool active_A = parent_A->is_active(i);
        bool active_B = parent_B->is_active(i);

        if (curand_uniform(rand_state) < p_A) {
            child->set_active(i, active_A);
        } else {
            child->set_active(i, active_B);
        }
    }

    // Inherit assembly biases (blend)
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        float blend = curand_uniform(rand_state);
        child->assembly_bias[i] = blend * parent_A->assembly_bias[i] +
                                  (1.0f - blend) * parent_B->assembly_bias[i];
    }

    // Resolve address conflicts in child
    resolve_address_conflicts(child);

    // Compute new hash
    child->structural_hash = compute_genome_hash(child);
}

__global__ void recombine_population_kernel(
    const Organism* parent_population,
    Organism* child_population,
    const int* parent_indices,  // [POPULATION_SIZE × 2] parent pairs
    int pop_size
) {
    int idx = GridStride::thread_id();
    if (idx < pop_size) {
        curandState local_state;
        curand_init(CURAND_SEED, idx + pop_size * 1000, 0, &local_state);

        int parent_A_idx = parent_indices[idx * 2];
        int parent_B_idx = parent_indices[idx * 2 + 1];

        const Organism* parent_A = &parent_population[parent_A_idx];
        const Organism* parent_B = &parent_population[parent_B_idx];
        Organism* child = &child_population[idx];

        // Use causal attribution as fitness weights
        recombine_asymmetric(
            &parent_A->genome,
            &parent_B->genome,
            &child->genome,
            parent_A->segment_activation_freq,
            parent_B->segment_activation_freq,
            &local_state
        );

        // Inherit parent IDs
        child->parent_a_id = parent_A->organism_id;
        child->parent_b_id = parent_B->organism_id;
    }
}

// ============================================================================
// Point Mutations (Per-Weight Level)
// ============================================================================

__global__ void apply_point_mutations_kernel(
    Genome* genomes,
    float mutation_rate,
    float mutation_scale,
    int n
) {
    int idx = GridStride::thread_id();
    if (idx < n) {
        curandState local_state;
        curand_init(CURAND_SEED, idx + n * 999, 0, &local_state);

        Genome* genome = &genomes[idx];

        for (int seg_idx = 0; seg_idx < NUM_SEGMENTS; seg_idx++) {
            if (!genome->is_active(seg_idx)) continue;

            GenomeSegment* seg = &genome->segments[seg_idx];

            for (int i = 0; i < SEGMENT_PAYLOAD_SIZE; i++) {
                if (curand_uniform(&local_state) < mutation_rate) {
                    seg->payload[i] += curand_normal(&local_state) * mutation_scale;
                    seg->payload[i] = clamp(seg->payload[i], -3.0f, 3.0f);
                }
            }
        }
    }
}
