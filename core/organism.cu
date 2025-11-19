#include "../config/constants.cuh"
#include "types.cuh"
#include "genome.cu"
#include "genome_mutations.cu"
#include "genome_variants.cu"
#include "flow_lenia.cu"
#include "../kernels/tensor_wmma.cu"
#include "../memory/stigmergy.cu"
#include "../memory/trace_encoding.cu"
#include "../metrics/fitness.cu"
#include "../lifecycle/selection.cu"
#include "../utils/tile_ops.cuh"
#include <cuda_runtime.h>

// Top-level organism orchestrator
// Manages complete evaluation pipeline: genome → CA → behavior → fitness

// Single organism evaluation (called per organism)
__global__ void evaluate_organism_kernel(
    Organism* organism,
    TraceEncoder* trace_encoder,
    StigmergicField* stigmergy,
    int organism_id,
    int num_timesteps,
    float dt
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Stage 1: Assemble CA weights from genome
    float* ca_weights;
    cudaMalloc(&ca_weights, GRID_SIZE * GRID_SIZE * NUM_HEADS * 256 * sizeof(float));

    dim3 grid_2d(GRID_SIZE / BLOCK_SIZE_2D, GRID_SIZE / BLOCK_SIZE_2D);
    dim3 block_2d(BLOCK_SIZE_2D, BLOCK_SIZE_2D);

    assemble_population_ca_weights_kernel<<<grid_2d, block_2d>>>(
        &organism->genome,
        ca_weights,
        stigmergy,
        1,
        organism_id
    );
    cudaDeviceSynchronize();

    // Stage 2: Initialize CA state
    float* ca_concentration;
    cudaMalloc(&ca_concentration, GRID_SIZE * GRID_SIZE * CHANNELS * sizeof(float));

    zero_memory_kernel<<<32, BLOCK_SIZE_1D>>>(ca_concentration, GRID_SIZE * GRID_SIZE * CHANNELS);

    // Seed initial pattern
    int seed_x = GRID_SIZE / 2;
    int seed_y = GRID_SIZE / 2;
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            int x = (seed_x + dx + GRID_SIZE) % GRID_SIZE;
            int y = (seed_y + dy + GRID_SIZE) % GRID_SIZE;
            int idx = y * GRID_SIZE + x;
            for (int c = 0; c < CHANNELS; c++) {
                ca_concentration[idx * CHANNELS + c] = (dx*dx + dy*dy < 5) ? 1.0f : 0.0f;
            }
        }
    }

    // Stage 3: Flow-Lenia simulation
    KernelSpec* kernels;
    cudaMalloc(&kernels, NUM_HEADS * sizeof(KernelSpec));

    initialize_kernels_from_genome_kernel<<<1, NUM_HEADS>>>(&organism->genome, kernels, NUM_HEADS);
    cudaDeviceSynchronize();

    float* affinity_map;
    float* flow_field_x;
    float* flow_field_y;
    float* next_concentration;

    cudaMalloc(&affinity_map, GRID_SIZE * GRID_SIZE * CHANNELS * sizeof(float));
    cudaMalloc(&flow_field_x, GRID_SIZE * GRID_SIZE * CHANNELS * sizeof(float));
    cudaMalloc(&flow_field_y, GRID_SIZE * GRID_SIZE * CHANNELS * sizeof(float));
    cudaMalloc(&next_concentration, GRID_SIZE * GRID_SIZE * CHANNELS * sizeof(float));

    float* flow_magnitude_trace;
    cudaMalloc(&flow_magnitude_trace, GRID_SIZE * GRID_SIZE * sizeof(float));

    for (int t = 0; t < num_timesteps; t++) {
        compute_affinity_map_kernel<<<grid_2d, block_2d>>>(
            ca_concentration,
            kernels,
            affinity_map,
            NUM_HEADS,
            10.0f
        );
        cudaDeviceSynchronize();

        compute_flow_field_kernel<<<grid_2d, block_2d>>>(
            affinity_map,
            ca_concentration,
            flow_field_x,
            flow_field_y,
            1.0f,
            2
        );
        cudaDeviceSynchronize();

        reintegration_tracking_kernel<<<grid_2d, block_2d>>>(
            ca_concentration,
            flow_field_x,
            flow_field_y,
            next_concentration,
            dt,
            0.5f
        );
        cudaDeviceSynchronize();

        float* temp = ca_concentration;
        ca_concentration = next_concentration;
        next_concentration = temp;

        if (trace_encoder->recording_enabled) {
            record_concentration_snapshot_kernel<<<32, BLOCK_SIZE_1D>>>(
                trace_encoder,
                organism_id,
                ca_concentration,
                t
            );

            record_flow_field_snapshot_kernel<<<32, BLOCK_SIZE_1D>>>(
                trace_encoder,
                organism_id,
                flow_field_x,
                flow_field_y,
                t
            );
        }

        for (int cell = threadIdx.x; cell < GRID_SIZE * GRID_SIZE; cell += blockDim.x) {
            float mag = 0.0f;
            for (int c = 0; c < CHANNELS; c++) {
                float fx = flow_field_x[cell * CHANNELS + c];
                float fy = flow_field_y[cell * CHANNELS + c];
                mag += sqrtf(fx * fx + fy * fy);
            }
            flow_magnitude_trace[cell] = mag / CHANNELS;
        }
    }

    // Stage 4: Extract behavioral coordinates
    float* behavioral_coords;
    cudaMalloc(&behavioral_coords, BEHAVIOR_DIM * sizeof(float));

    compute_behavioral_coordinates_kernel<<<1, 256>>>(
        ca_concentration,
        stigmergy->structural_layer,
        flow_magnitude_trace,
        behavioral_coords,
        organism->rand_state,
        organism_id
    );
    cudaDeviceSynchronize();

    cudaMemcpy(organism->behavioral_coords, behavioral_coords, BEHAVIOR_DIM * sizeof(float), cudaMemcpyDeviceToDevice);

    // Stage 5: Compute fitness
    compute_fitness_kernel<<<1, 1>>>(
        ca_concentration,
        &organism->genome,
        stigmergy,
        &organism->fitness,
        GRID_SIZE * GRID_SIZE
    );
    cudaDeviceSynchronize();

    // Store trace endpoint
    if (trace_encoder->recording_enabled) {
        ExecutionTrace* trace = &trace_encoder->traces[organism_id];
        cudaMemcpy(trace->behavioral_coordinates, behavioral_coords, BEHAVIOR_DIM * sizeof(float), cudaMemcpyDeviceToDevice);
        trace->fitness = organism->fitness;
    }

    // Cleanup
    cudaFree(ca_weights);
    cudaFree(ca_concentration);
    cudaFree(kernels);
    cudaFree(affinity_map);
    cudaFree(flow_field_x);
    cudaFree(flow_field_y);
    cudaFree(next_concentration);
    cudaFree(flow_magnitude_trace);
    cudaFree(behavioral_coords);
}

__global__ void evaluate_population_batch_kernel(
    Organism* population,
    int population_size,
    TraceEncoder* trace_encoder,
    StigmergicField* stigmergy,
    int num_timesteps,
    float dt
) {
    int organism_id = blockIdx.x;

    if (organism_id >= population_size) return;

    if (threadIdx.x == 0) {
        evaluate_organism_kernel<<<1, 1>>>(
            &population[organism_id],
            trace_encoder,
            stigmergy,
            organism_id,
            num_timesteps,
            dt
        );
    }
}

// Create offspring from parents
__global__ void create_offspring_kernel(
    const Genome* parent_a,
    const Genome* parent_b,
    Genome* offspring,
    const RegulatoryState* regulatory,
    curandState* rand_state
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Recombination
    if (curand_uniform(rand_state) < regulatory->recombination_rate) {
        // Asymmetric recombination using causal attribution
        float fitness_a[NUM_SEGMENTS];
        float fitness_b[NUM_SEGMENTS];

        for (int seg = 0; seg < NUM_SEGMENTS; seg++) {
            fitness_a[seg] = parent_a->segments[seg].causal_attribution;
            fitness_b[seg] = parent_b->segments[seg].causal_attribution;
        }

        recombine_asymmetric(parent_a, parent_b, offspring, fitness_a, fitness_b, rand_state);
    } else {
        // Clone parent A
        *offspring = *parent_a;
    }

    // Mutation
    for (int seg = 0; seg < NUM_SEGMENTS; seg++) {
        GenomeSegment* segment = &offspring->segments[seg];

        if (can_mutate_segment(segment, regulatory->lifecycle_crisis, rand_state)) {
            // Point mutations scaled by regulatory multiplier
            float mutation_strength = 0.1f * regulatory->mutation_rate_multiplier;

            for (int w = 0; w < SEGMENT_PAYLOAD_SIZE; w++) {
                if (curand_uniform(rand_state) < 0.05f) {  // 5% per weight
                    segment->payload[w] += curand_normal(rand_state) * mutation_strength;
                    segment->payload[w] = clamp(segment->payload[w], -10.0f, 10.0f);
                }
            }
        }

        // Structural mutations
        if (curand_uniform(rand_state) < regulatory->structural_mutation_rate) {
            int mutation_type = curand(rand_state) % 3;

            if (mutation_type == 0) {
                // Transposition
                int target_seg = curand(rand_state) % NUM_SEGMENTS;
                transpose_segment(offspring, seg, target_seg, false, rand_state);
            } else if (mutation_type == 1 && NUM_SEGMENTS > 1) {
                // Excision (mark inactive)
                segment->address_tag = 0xFFFF;
            } else {
                // Address tag mutation
                segment->address_tag = curand(rand_state) & 0xFFFF;
            }
        }
    }

    // Update lineage depth
    offspring->lineage_depth = fmaxf(parent_a->lineage_depth, parent_b->lineage_depth) + 1;
}

__global__ void reproduce_population_kernel(
    const Organism* population,
    Organism* next_generation,
    const Archive* archive,
    const PopulationMetrics* metrics,
    const RegulatoryState* regulatory,
    int population_size,
    curandState* rand_states
) {
    int offspring_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (offspring_id >= population_size) return;

    Genome parent_a, parent_b;

    select_parents_hybrid_kernel<<<1, 1>>>(
        population,
        archive,
        metrics,
        &parent_a,
        &parent_b,
        population_size,
        rand_states,
        offspring_id
    );
    cudaDeviceSynchronize();

    create_offspring_kernel<<<1, 1>>>(
        &parent_a,
        &parent_b,
        &next_generation[offspring_id].genome,
        regulatory,
        &rand_states[offspring_id]
    );
    cudaDeviceSynchronize();

    next_generation[offspring_id].fitness = -FLT_MAX;
}
