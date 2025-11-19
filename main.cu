#include "core/runtime.cu"
#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int population_size = 64;
    int num_generations = 1000;

    if (argc > 1) {
        population_size = atoi(argv[1]);
    }
    if (argc > 2) {
        num_generations = atoi(argv[2]);
    }

    printf("=== GeneSlime: Genomic Flow-Lenia Evolution ===\n");
    printf("Population: %d\n", population_size);
    printf("Generations: %d\n\n", num_generations);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("CUDA Cores: %d SMs\n\n", prop.multiProcessorCount);

    Runtime runtime;
    runtime_allocate(&runtime, population_size, num_generations);
    runtime_init_population(&runtime);
    runtime_evolve(&runtime);
    runtime_free(&runtime);

    printf("\nEvolution complete.\n");

    return 0;
}
