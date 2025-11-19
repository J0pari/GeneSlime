#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "types.cuh"
#include "../utils/tile_ops.cuh"
#include "../kernels/utils.cu"

// ============================================================================
// Context-Dependent Genome Expansion
// ============================================================================

__device__ void expand_segment_contextual(
    const GenomeSegment* segment,
    float* target_weights,
    int grid_x,
    int grid_y,
    const float* local_field_state,
    int generation,
    curandState* rand_state
) {
    // Decode address tag
    int target_x = segment->get_grid_x();
    int target_y = segment->get_grid_y();
    int head = segment->get_head_index();

    // Spatial context: position-dependent scaling
    float spatial_scale = 1.0f;
    if (grid_x < GRID_SIZE / 2) {
        spatial_scale = 0.8f; // Left half: conservative
    } else {
        spatial_scale = 1.2f; // Right half: amplified
    }

    // Field state context: mass-dependent modulation
    float mass = local_field_state[0]; // Assume first channel is mass
    float field_scale = (mass > 0.5f) ? 0.9f : 1.1f;

    // Developmental timing: generational drift
    float temporal_scale = 1.0f + 0.01f * (generation % 100);

    // Distance from target: local vs distal expression
    int dx = abs(grid_x - target_x);
    int dy = abs(grid_y - target_y);
    float distance = sqrtf((float)(dx * dx + dy * dy));
    float distance_scale = expf(-distance / 10.0f); // Gaussian decay

    // Combined contextual modulation
    float total_scale = spatial_scale * field_scale * temporal_scale * distance_scale;

    // Expand payload with context
    #pragma unroll 8
    for (int i = 0; i < SEGMENT_PAYLOAD_SIZE; i++) {
        target_weights[i] = segment->payload[i] * total_scale;
    }

    // Add stochastic noise based on permission level
    if (segment->permission_level >= LEVEL_2_PARAMETRIC) {
        float noise_scale = (segment->permission_level == LEVEL_3_REGULATORY) ? 0.1f : 0.05f;
        for (int i = 0; i < SEGMENT_PAYLOAD_SIZE; i++) {
            target_weights[i] += curand_normal(rand_state) * noise_scale;
        }
    }
}

// ============================================================================
// Address Resolution and Priority-Based Assembly
// ============================================================================

__device__ void resolve_address_conflicts(Genome* genome) {
    // Build address -> segment mapping
    int address_map[GRID_SIZE * GRID_SIZE * NUM_HEADS];
    int priority_map[GRID_SIZE * GRID_SIZE * NUM_HEADS];

    // Initialize with -1 (no segment)
    for (int i = 0; i < GRID_SIZE * GRID_SIZE * NUM_HEADS; i++) {
        address_map[i] = -1;
        priority_map[i] = -1;
    }

    // Scan all active segments
    for (int seg_idx = 0; seg_idx < NUM_SEGMENTS; seg_idx++) {
        if (!genome->is_active(seg_idx)) continue;

        GenomeSegment* seg = &genome->segments[seg_idx];
        int gx = seg->get_grid_x();
        int gy = seg->get_grid_y();
        int head = seg->get_head_index();

        // Compute linear address
        int addr = (gy * GRID_SIZE + gx) * NUM_HEADS + head;

        // Check for conflict
        if (address_map[addr] == -1) {
            // No conflict, assign
            address_map[addr] = seg_idx;
            priority_map[addr] = seg->priority;
        } else {
            // Conflict detected: resolve by priority
            int existing_seg = address_map[addr];
            if (seg->priority > genome->segments[existing_seg].priority) {
                // New segment wins: shadow old segment
                genome->set_active(existing_seg, false);
                address_map[addr] = seg_idx;
                priority_map[addr] = seg->priority;
            } else {
                // Old segment wins: deactivate new segment
                genome->set_active(seg_idx, false);
            }
        }
    }
}

__global__ void resolve_address_conflicts_kernel(Genome* genomes, int n) {
    int idx = GridStride::thread_id();
    if (idx < n) {
        resolve_address_conflicts(&genomes[idx]);
    }
}

// ============================================================================
// Genome Assembly: Expand Segments into CA Weights
// ============================================================================

__device__ void assemble_ca_weights_from_genome(
    const Genome* genome,
    half* perception_weights,      // [NUM_HEADS × CHANNELS × HIDDEN_DIM]
    half* interaction_weights,     // [NUM_HEADS × HIDDEN_DIM × HIDDEN_DIM]
    half* value_weights,           // [NUM_HEADS × HIDDEN_DIM × HEAD_DIM]
    const StigmergicField* stigmergy,
    int generation,
    curandState* rand_state
) {
    // Initialize all weights to small random values
    for (int i = 0; i < NUM_HEADS * CHANNELS * HIDDEN_DIM; i++) {
        perception_weights[i] = __float2half(curand_normal(rand_state) * 0.1f);
    }

    // Iterate over active segments and expand
    for (int seg_idx = 0; seg_idx < NUM_SEGMENTS; seg_idx++) {
        if (!genome->is_active(seg_idx)) continue;

        const GenomeSegment* seg = &genome->segments[seg_idx];
        int gx = seg->get_grid_x();
        int gy = seg->get_grid_y();
        int head = seg->get_head_index();

        // Read local field state for context
        int cell_idx = gy * GRID_SIZE + gx;
        float local_field[NUM_STIGMERGY_LAYERS];
        local_field[0] = stigmergy->fast_layer[cell_idx];
        local_field[1] = stigmergy->medium_layer[cell_idx];
        local_field[2] = stigmergy->slow_layer[cell_idx];
        local_field[3] = stigmergy->structural_layer[cell_idx];

        // Expand segment with context
        float expanded[SEGMENT_PAYLOAD_SIZE];
        expand_segment_contextual(seg, expanded, gx, gy, local_field, generation, rand_state);

        // Map expanded weights to CA weights based on permission level
        if (seg->permission_level == LEVEL_0_PROTECTED || seg->permission_level == LEVEL_1_STRUCTURAL) {
            // Core infrastructure: perception weights
            int offset = head * CHANNELS * HIDDEN_DIM;
            for (int i = 0; i < SEGMENT_PAYLOAD_SIZE && i < CHANNELS * HIDDEN_DIM; i++) {
                perception_weights[offset + i] = __float2half(expanded[i]);
            }
        } else if (seg->permission_level == LEVEL_2_PARAMETRIC) {
            // Parametric weights: interaction matrix
            int offset = head * HIDDEN_DIM * HIDDEN_DIM;
            for (int i = 0; i < SEGMENT_PAYLOAD_SIZE && i < HIDDEN_DIM * HIDDEN_DIM; i++) {
                interaction_weights[offset + i] = __float2half(expanded[i]);
            }
        } else if (seg->permission_level == LEVEL_3_REGULATORY) {
            // Regulatory: value projection
            int offset = head * HIDDEN_DIM * HEAD_DIM;
            for (int i = 0; i < SEGMENT_PAYLOAD_SIZE && i < HIDDEN_DIM * HEAD_DIM; i++) {
                value_weights[offset + i] = __float2half(expanded[i]);
            }
        }
    }
}

__global__ void assemble_population_ca_weights_kernel(
    Organism* population,
    int pop_size,
    int generation
) {
    int idx = GridStride::thread_id();
    if (idx < pop_size) {
        Organism* org = &population[idx];
        assemble_ca_weights_from_genome(
            &org->genome,
            org->ca_state.perception_weights,
            org->ca_state.interaction_weights,
            org->ca_state.value_weights,
            org->stigmergy,
            generation,
            &org->rand_state
        );
    }
}

// ============================================================================
// Structural Operations: Excision, Insertion, Transposition
// ============================================================================

__device__ void excise_segment(Genome* genome, int segment_idx) {
    // Zero payload
    for (int i = 0; i < SEGMENT_PAYLOAD_SIZE; i++) {
        genome->segments[segment_idx].payload[i] = 0.0f;
    }

    // Mark inactive
    genome->set_active(segment_idx, false);

    // Preserve address tag for future insertion (leaves spacer)
}

__device__ bool insert_segment(Genome* genome, int target_idx, const GenomeSegment* new_segment) {
    if (genome->is_active(target_idx)) {
        // Existing segment active: check priority
        if (new_segment->priority > genome->segments[target_idx].priority) {
            // New segment shadows old (multi-allelic encoding)
            genome->segments[target_idx] = *new_segment;
            return true;
        } else {
            // Insertion fails, old segment retained
            return false;
        }
    } else {
        // Spacer location, insert freely
        genome->segments[target_idx] = *new_segment;
        genome->set_active(target_idx, true);
        return true;
    }
}

__device__ bool transpose_segment(
    Genome* genome,
    int source_idx,
    int dest_idx,
    bool copy,  // true = duplication, false = move
    curandState* rand_state
) {
    if (!genome->segments[source_idx].mobility_flag) {
        return false; // Segment not mobile
    }

    GenomeSegment seg = genome->segments[source_idx];

    // Attempt insertion
    bool success = insert_segment(genome, dest_idx, &seg);

    if (success && !copy) {
        // Move: excise source
        excise_segment(genome, source_idx);
    }

    return success;
}

__global__ void transpose_segments_kernel(
    Genome* genomes,
    int n,
    float transposition_rate
) {
    int idx = GridStride::thread_id();
    if (idx < n) {
        Genome* genome = &genomes[idx];
        curandState local_state;
        curand_init(CURAND_SEED, idx, 0, &local_state);

        // Attempt transposition for each active mobile segment
        for (int i = 0; i < NUM_SEGMENTS; i++) {
            if (!genome->is_active(i)) continue;
            if (!genome->segments[i].mobility_flag) continue;

            if (curand_uniform(&local_state) < transposition_rate) {
                // Select random destination
                int dest = curand(&local_state) % NUM_SEGMENTS;

                // 50% chance of copy vs move
                bool copy = curand_uniform(&local_state) < 0.5f;

                transpose_segment(genome, i, dest, copy, &local_state);
            }
        }
    }
}

// ============================================================================
// Initialization: Random Genome Generation
// ============================================================================

__global__ void initialize_random_genomes_kernel(Genome* genomes, int n, unsigned long long seed) {
    int idx = GridStride::thread_id();
    if (idx < n) {
        curandState local_state;
        curand_init(seed, idx, 0, &local_state);

        Genome* genome = &genomes[idx];

        // Initialize all segments as active
        genome->active_mask = 0xFFFF; // All 16 segments active

        for (int seg_idx = 0; seg_idx < NUM_SEGMENTS; seg_idx++) {
            GenomeSegment* seg = &genome->segments[seg_idx];

            // Random payload
            for (int i = 0; i < SEGMENT_PAYLOAD_SIZE; i++) {
                seg->payload[i] = curand_normal(&local_state) * 0.5f;
            }

            // Assign address tags
            int gx = curand(&local_state) % GRID_SIZE;
            int gy = curand(&local_state) % GRID_SIZE;
            int head = seg_idx % NUM_HEADS;
            int variant = 0;
            seg->set_address(gx, gy, head, variant);

            // Permission levels: stratified
            if (seg_idx < 4) {
                seg->permission_level = LEVEL_0_PROTECTED;
            } else if (seg_idx < 8) {
                seg->permission_level = LEVEL_1_STRUCTURAL;
            } else if (seg_idx < 12) {
                seg->permission_level = LEVEL_2_PARAMETRIC;
            } else {
                seg->permission_level = LEVEL_3_REGULATORY;
            }

            // Priority and mobility
            seg->priority = curand(&local_state) % 256;
            seg->mobility_flag = (seg->permission_level >= LEVEL_2_PARAMETRIC);

            // Initialize stochastic variants
            for (int v = 0; v < SEGMENT_PAYLOAD_SIZE; v++) {
                seg->stochastic.variant_A.expansion[v] = seg->payload[v] * 0.9f;
                seg->stochastic.variant_B.expansion[v] = seg->payload[v] * 1.2f;
                seg->stochastic.variant_C.expansion[v] = curand_normal(&local_state) * 0.8f;
            }

            seg->stochastic.variant_A.base_probability = VARIANT_A_BASE_PROB;
            seg->stochastic.variant_B.base_probability = VARIANT_B_BASE_PROB;
            seg->stochastic.variant_C.base_probability = VARIANT_C_BASE_PROB;

            // Initialize traces
            seg->expression_frequency = 0.0f;
            seg->causal_attribution = 0.0f;
        }

        // Initialize assembly biases
        for (int i = 0; i < NUM_SEGMENTS; i++) {
            genome->assembly_bias[i] = 1.0f;
        }

        // Compute structural hash
        genome->structural_hash = compute_genome_hash(genome);
    }
}
