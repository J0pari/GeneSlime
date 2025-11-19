#include "../config/constants.cuh"
#include "../core/types.cuh"
#include "../memory/archive.cu"

__global__ void batch_archive_insertion_kernel(
    Archive* archive,
    Organism* population,
    int population_size,
    uint32_t generation
) {
    int organism_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (organism_id >= population_size) return;

    bool success;

    archive_insertion_kernel<<<1, 1>>>(
        archive,
        &population[organism_id].genome,
        population[organism_id].fitness,
        population[organism_id].behavioral_coords,
        generation,
        &success
    );
}
