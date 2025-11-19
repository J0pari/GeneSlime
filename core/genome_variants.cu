#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "types.cuh"
#include "../utils/tile_ops.cuh"

// ============================================================================
// Stochastic Variant Selection with Context-Dependent Probabilities
// ============================================================================

__device__ void select_and_expand_variant(
    StochasticSegment* stochastic_seg,
    float* target_weights,
    float local_mass,
    float coherence,
    float archive_saturation,
    curandState* rand_state
) {
    // Compute variant probabilities based on context
    float p_A = stochastic_seg->variant_A.base_probability;
    float p_B = stochastic_seg->variant_B.base_probability;
    float p_C = stochastic_seg->variant_C.base_probability;

    // Stress increases exploratory/disruptive variants
    if (coherence < CRISIS_COHERENCE_THRESHOLD) {
        p_A *= 0.7f;
        p_B *= 1.5f;
        p_C *= 2.0f;
    }

    // Low archive saturation increases exploration
    if (archive_saturation < CRISIS_SATURATION_THRESHOLD) {
        p_B *= 1.3f;
        p_C *= 1.5f;
    }

    // Normalize probabilities
    float total = p_A + p_B + p_C;
    p_A /= total;
    p_B /= total;
    p_C /= total;

    // Sample variant
    float r = curand_uniform(rand_state);
    const SegmentVariant* selected;

    if (r < p_A) {
        selected = &stochastic_seg->variant_A;
    } else if (r < p_A + p_B) {
        selected = &stochastic_seg->variant_B;
    } else {
        selected = &stochastic_seg->variant_C;
    }

    // Expand selected variant
    #pragma unroll 8
    for (int i = 0; i < SEGMENT_PAYLOAD_SIZE; i++) {
        target_weights[i] = selected->expansion[i];
    }
}

// ============================================================================
// Assembly Probability Modulation (Population-Level Feedback)
// ============================================================================

__device__ void modulate_assembly_probabilities(
    StochasticSegment* segment,
    const PopulationMetrics* metrics
) {
    float mean_saturation = metrics->get_mean_saturation();

    // Underpopulated niches → increase exploration
    if (mean_saturation < CRISIS_SATURATION_THRESHOLD) {
        segment->variant_A.base_probability *= 0.7f;
        segment->variant_B.base_probability *= 1.5f;
        segment->variant_C.base_probability *= 2.0f;
    }

    // Population converged → amplify structural shuffling
    if (metrics->fitness_variance < CRISIS_VARIANCE_THRESHOLD) {
        // Increase disruptive variant probability
        segment->variant_C.base_probability *= 1.5f;
    }

    // Population not learning → conservative variants
    if (metrics->coherence_mean < CRISIS_COHERENCE_THRESHOLD) {
        // Prefer conservative variant
        segment->variant_A.base_probability *= 1.3f;
        segment->variant_B.base_probability *= 0.8f;
        segment->variant_C.base_probability *= 0.5f;
    }

    // Renormalize
    float total = segment->variant_A.base_probability +
                  segment->variant_B.base_probability +
                  segment->variant_C.base_probability;

    segment->variant_A.base_probability /= total;
    segment->variant_B.base_probability /= total;
    segment->variant_C.base_probability /= total;
}

__global__ void modulate_population_assembly_kernel(
    Organism* population,
    const PopulationMetrics* metrics,
    int pop_size
) {
    int idx = GridStride::thread_id();
    if (idx < pop_size) {
        Organism* org = &population[idx];

        // Modulate each segment's variant probabilities
        for (int seg_idx = 0; seg_idx < NUM_SEGMENTS; seg_idx++) {
            if (org->genome.is_active(seg_idx)) {
                modulate_assembly_probabilities(
                    &org->genome.segments[seg_idx].stochastic,
                    metrics
                );
            }
        }
    }
}

// ============================================================================
// Variant Diversity Tracking
// ============================================================================

__global__ void track_variant_selection_kernel(
    Organism* population,
    int* variant_counts,  // [NUM_SEGMENTS × NUM_VARIANTS]
    int pop_size
) {
    int idx = GridStride::thread_id();
    if (idx < pop_size) {
        Organism* org = &population[idx];

        for (int seg_idx = 0; seg_idx < NUM_SEGMENTS; seg_idx++) {
            if (!org->genome.is_active(seg_idx)) continue;

            // Determine which variant was selected (highest probability)
            const StochasticSegment* stoch = &org->genome.segments[seg_idx].stochastic;
            int selected_variant = 0;

            if (stoch->variant_B.base_probability > stoch->variant_A.base_probability &&
                stoch->variant_B.base_probability > stoch->variant_C.base_probability) {
                selected_variant = 1;
            } else if (stoch->variant_C.base_probability > stoch->variant_A.base_probability &&
                       stoch->variant_C.base_probability > stoch->variant_B.base_probability) {
                selected_variant = 2;
            }

            // Atomically increment count
            atomicAdd(&variant_counts[seg_idx * NUM_VARIANTS + selected_variant], 1);
        }
    }
}
