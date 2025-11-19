#include "../config/constants.cuh"
#include "../debug/auto_trace.cuh"
#include "types.cuh"
#include "organism.cu"
#include "../memory/archive.cu"
#include "../metrics/population.cu"
#include "../lifecycle/regulatory.cu"
#include "../kernels/utils.cu"
#include "../kernels/archive_batch.cu"
#include <cuda_runtime.h>
#include <stdio.h>

// Host-side runtime for evolutionary loop

struct Runtime {
    // Device pointers
    Organism* d_population;
    Organism* d_next_generation;
    Archive* d_archive;
    TraceEncoder* d_trace_encoder;
    StigmergicField* d_stigmergy;
    PopulationMetrics* d_metrics;
    RegulatoryState* d_regulatory;
    curandState* d_rand_states;

    // Host parameters
    int population_size;
    int num_generations;
    int num_timesteps;
    float dt;
    uint32_t current_generation;
};

// Allocate device memory for runtime
void runtime_allocate(Runtime* runtime, int population_size, int num_generations) {
    printf("[RUNTIME] Allocating device memory...\n");

    runtime->population_size = population_size;
    runtime->num_generations = num_generations;
    runtime->num_timesteps = 100;
    runtime->dt = 0.1f;
    runtime->current_generation = 0;

    CHECK_CUDA(cudaMalloc(&runtime->d_population, population_size * sizeof(Organism)));
    VALIDATE_DEVICE_PTR(runtime->d_population);

    CHECK_CUDA(cudaMalloc(&runtime->d_next_generation, population_size * sizeof(Organism)));
    VALIDATE_DEVICE_PTR(runtime->d_next_generation);

    CHECK_CUDA(cudaMalloc(&runtime->d_archive, sizeof(Archive)));
    VALIDATE_DEVICE_PTR(runtime->d_archive);

    Archive h_archive;
    h_archive.num_cells = 100;

    CHECK_CUDA(cudaMalloc(&h_archive.cells, h_archive.num_cells * sizeof(VoronoiCell)));
    VALIDATE_DEVICE_PTR(h_archive.cells);

    CHECK_CUDA(cudaMalloc(&h_archive.behavioral_bounds_min, BEHAVIOR_DIM * sizeof(float)));
    VALIDATE_DEVICE_PTR(h_archive.behavioral_bounds_min);

    CHECK_CUDA(cudaMalloc(&h_archive.behavioral_bounds_max, BEHAVIOR_DIM * sizeof(float)));
    VALIDATE_DEVICE_PTR(h_archive.behavioral_bounds_max);

    h_archive.total_insertions = 0;
    h_archive.successful_insertions = 0;

    // Set behavioral space bounds
    float h_bounds_min[BEHAVIOR_DIM];
    float h_bounds_max[BEHAVIOR_DIM];
    for (int d = 0; d < BEHAVIOR_DIM; d++) {
        h_bounds_min[d] = 0.0f;
        h_bounds_max[d] = 1.0f;
    }
    cudaMemcpy(h_archive.behavioral_bounds_min, h_bounds_min, BEHAVIOR_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(h_archive.behavioral_bounds_max, h_bounds_max, BEHAVIOR_DIM * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(runtime->d_archive, &h_archive, sizeof(Archive), cudaMemcpyHostToDevice);

    // Allocate trace encoder
    cudaMalloc(&runtime->d_trace_encoder, sizeof(TraceEncoder));

    // Allocate stigmergic field
    cudaMalloc(&runtime->d_stigmergy, sizeof(StigmergicField));
    StigmergicField h_stigmergy;

    cudaMalloc(&h_stigmergy.fast_layer, GRID_SIZE * GRID_SIZE * sizeof(float));
    cudaMalloc(&h_stigmergy.medium_layer, GRID_SIZE * GRID_SIZE * sizeof(float));
    cudaMalloc(&h_stigmergy.slow_layer, GRID_SIZE * GRID_SIZE * sizeof(float));
    cudaMalloc(&h_stigmergy.structural_layer, GRID_SIZE * GRID_SIZE * sizeof(float));

    h_stigmergy.decay_rates[0] = DECAY_FAST;
    h_stigmergy.decay_rates[1] = DECAY_MEDIUM;
    h_stigmergy.decay_rates[2] = DECAY_SLOW;
    h_stigmergy.decay_rates[3] = 0.9999f;

    for (int i = 0; i < NUM_STIGMERGY_LAYERS; i++) {
        h_stigmergy.write_thresholds[i] = 0.1f;
    }

    cudaMemcpy(runtime->d_stigmergy, &h_stigmergy, sizeof(StigmergicField), cudaMemcpyHostToDevice);

    // Allocate metrics
    cudaMalloc(&runtime->d_metrics, sizeof(PopulationMetrics));

    // Allocate regulatory state
    cudaMalloc(&runtime->d_regulatory, sizeof(RegulatoryState));

    // Allocate RNG states
    cudaMalloc(&runtime->d_rand_states, population_size * 256 * sizeof(curandState));

    // Initialize RNG
    init_curand_states<<<32, 256>>>(runtime->d_rand_states, population_size * 256, 42);
    cudaDeviceSynchronize();

    // Initialize archive with CVT
    Archive* d_archive_ptr = runtime->d_archive;
    initialize_archive_kernel<<<1, 256>>>(
        d_archive_ptr,
        h_archive.num_cells,
        runtime->d_rand_states,
        10  // CVT iterations
    );
    cudaDeviceSynchronize();

    // Initialize regulatory state
    init_regulatory_state_kernel<<<1, 1>>>(runtime->d_regulatory);
    cudaDeviceSynchronize();

    printf("Runtime allocated: %d organisms, %d generations\n", population_size, num_generations);
}

// Initialize population with random genomes
void runtime_init_population(Runtime* runtime) {
    init_organism_rng<<<32, 256>>>(runtime->d_population, runtime->population_size, 123456);
    cudaDeviceSynchronize();

    // Initialize genomes with random weights
    for (int i = 0; i < runtime->population_size; i++) {
        Organism h_organism;

        for (int seg = 0; seg < NUM_SEGMENTS; seg++) {
            h_organism.genome.segments[seg].address_tag = (rand() & 0xFFFF);
            h_organism.genome.segments[seg].permission_level = seg % 4;
            h_organism.genome.segments[seg].priority = seg;
            h_organism.genome.segments[seg].mobility_flag = (seg % 3 == 0);
            h_organism.genome.segments[seg].expression_frequency = 1.0f;
            h_organism.genome.segments[seg].causal_attribution = 0.5f;

            for (int w = 0; w < SEGMENT_PAYLOAD_SIZE; w++) {
                h_organism.genome.segments[seg].payload[w] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
            }
        }

        h_organism.genome.lineage_depth = 0;
        h_organism.fitness = -FLT_MAX;

        cudaMemcpy(&runtime->d_population[i], &h_organism, sizeof(Organism), cudaMemcpyHostToDevice);
    }

    printf("Population initialized with random genomes\n");
}

// Run single generation
void runtime_step_generation(Runtime* runtime) {
    // Evaluate population
    evaluate_population_batch_kernel<<<runtime->population_size, 1>>>(
        runtime->d_population,
        runtime->population_size,
        runtime->d_trace_encoder,
        runtime->d_stigmergy,
        runtime->num_timesteps,
        runtime->dt
    );
    cudaDeviceSynchronize();

    int threads = 256;
    int blocks = (runtime->population_size + threads - 1) / threads;

    batch_archive_insertion_kernel<<<blocks, threads>>>(
        runtime->d_archive,
        runtime->d_population,
        runtime->population_size,
        runtime->current_generation
    );
    cudaDeviceSynchronize();

    // Compute population metrics
    float* d_fitness_values;
    cudaMalloc(&d_fitness_values, runtime->population_size * sizeof(float));

    for (int i = 0; i < runtime->population_size; i++) {
        Organism h_organism;
        cudaMemcpy(&h_organism, &runtime->d_population[i], sizeof(Organism), cudaMemcpyDeviceToHost);
        cudaMemcpy(&d_fitness_values[i], &h_organism.fitness, sizeof(float), cudaMemcpyHostToDevice);
    }

    compute_population_metrics_kernel<<<1, 1>>>(
        runtime->d_archive,
        runtime->d_population,
        d_fitness_values,
        runtime->population_size,
        runtime->current_generation,
        runtime->d_metrics
    );
    cudaDeviceSynchronize();

    cudaFree(d_fitness_values);

    // Update regulatory state
    update_regulatory_state_kernel<<<1, 1>>>(
        runtime->d_metrics,
        runtime->d_archive,
        runtime->d_regulatory,
        runtime->current_generation,
        runtime->num_generations
    );
    cudaDeviceSynchronize();

    // Reproduce to create next generation
    int threads = 256;
    int blocks = (runtime->population_size + threads - 1) / threads;

    reproduce_population_kernel<<<blocks, threads>>>(
        runtime->d_population,
        runtime->d_next_generation,
        runtime->d_archive,
        runtime->d_metrics,
        runtime->d_regulatory,
        runtime->population_size,
        runtime->d_rand_states
    );
    cudaDeviceSynchronize();

    // Swap populations
    Organism* temp = runtime->d_population;
    runtime->d_population = runtime->d_next_generation;
    runtime->d_next_generation = temp;

    runtime->current_generation++;
}

// Print generation stats
void runtime_print_stats(Runtime* runtime) {
    // Copy metrics to host
    PopulationMetrics h_metrics;
    cudaMemcpy(&h_metrics, runtime->d_metrics, sizeof(PopulationMetrics), cudaMemcpyDeviceToHost);

    // Find best fitness
    float best_fitness = -FLT_MAX;
    float avg_fitness = 0.0f;

    for (int i = 0; i < runtime->population_size; i++) {
        Organism h_organism;
        cudaMemcpy(&h_organism, &runtime->d_population[i], sizeof(Organism), cudaMemcpyDeviceToHost);

        best_fitness = fmaxf(best_fitness, h_organism.fitness);
        avg_fitness += h_organism.fitness;
    }
    avg_fitness /= runtime->population_size;

    printf("Gen %d | Best: %.4f | Avg: %.4f | Archive: %.2f%% | Diversity: %.4f | Stagnation: %.1f\n",
           runtime->current_generation,
           best_fitness,
           avg_fitness,
           h_metrics.archive_saturation * 100.0f,
           h_metrics.genome_diversity,
           h_metrics.stagnation_score);
}

// Run full evolution
void runtime_evolve(Runtime* runtime) {
    printf("Starting evolution for %d generations...\n", runtime->num_generations);

    for (int gen = 0; gen < runtime->num_generations; gen++) {
        runtime_step_generation(runtime);

        if (gen % 10 == 0) {
            runtime_print_stats(runtime);
        }
    }

    printf("\nEvolution complete!\n");
    runtime_print_stats(runtime);
}

// Cleanup
void runtime_free(Runtime* runtime) {
    // Free archive internals
    Archive h_archive;
    cudaMemcpy(&h_archive, runtime->d_archive, sizeof(Archive), cudaMemcpyDeviceToHost);
    cudaFree(h_archive.cells);
    cudaFree(h_archive.behavioral_bounds_min);
    cudaFree(h_archive.behavioral_bounds_max);

    // Free stigmergy internals
    StigmergicField h_stigmergy;
    cudaMemcpy(&h_stigmergy, runtime->d_stigmergy, sizeof(StigmergicField), cudaMemcpyDeviceToHost);
    cudaFree(h_stigmergy.fast_layer);
    cudaFree(h_stigmergy.medium_layer);
    cudaFree(h_stigmergy.slow_layer);
    cudaFree(h_stigmergy.structural_layer);

    // Free main structures
    cudaFree(runtime->d_population);
    cudaFree(runtime->d_next_generation);
    cudaFree(runtime->d_archive);
    cudaFree(runtime->d_trace_encoder);
    cudaFree(runtime->d_stigmergy);
    cudaFree(runtime->d_metrics);
    cudaFree(runtime->d_regulatory);
    cudaFree(runtime->d_rand_states);

    printf("Runtime freed\n");
}
