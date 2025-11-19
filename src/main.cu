#include "debug/kernel_trace.cu"
#include "debug/param_validator.cu"
#include "core/runtime.cu"
#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char** argv) {
    init_kernel_trace();

    int population_size = 64;
    int num_generations = 1000;

    if (argc > 1) {
        population_size = atoi(argv[1]);
    }
    if (argc > 2) {
        num_generations = atoi(argv[2]);
    }

    VALIDATE_INT_RANGE(population_size, 1, MAX_POPULATION_SIZE);
    VALIDATE_INT_RANGE(num_generations, 1, 1000000);

    printf("=== GeneSlime: Genomic Flow-Lenia Evolution ===\n");
    printf("Population: %d\n", population_size);
    printf("Generations: %d\n\n", num_generations);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("CUDA Cores: %d SMs\n", prop.multiProcessorCount);
    printf("Max threads/block: %d\n", prop.maxThreadsPerBlock);
    printf("Max shared mem/block: %zu KB\n\n", prop.sharedMemPerBlock / 1024);

    PRINT_STRUCT(Organism);
    PRINT_STRUCT(Genome);
    PRINT_STRUCT(GenomeSegment);
    printf("\n");

    Runtime runtime;
    runtime_allocate(&runtime, population_size, num_generations);
    runtime_init_population(&runtime);
    runtime_evolve(&runtime);
    runtime_free(&runtime);

    printf("\nEvolution complete.\n");

    return 0;
}
