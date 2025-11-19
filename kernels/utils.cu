#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../core/types.cuh"
#include "../utils/tile_ops.cuh"

// ============================================================================
// RNG Initialization
// ============================================================================

__global__ void init_curand_states(curandState* states, int n, unsigned long long seed) {
    int idx = GridStride::thread_id();
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void init_organism_rng(Organism* population, int pop_size, unsigned long long seed) {
    int idx = GridStride::thread_id();
    if (idx < pop_size) {
        curand_init(seed, idx, 0, &population[idx].rand_state);
    }
}

// ============================================================================
// Hash Functions (for genome deduplication)
// ============================================================================

__device__ __forceinline__ uint64_t hash_combine(uint64_t seed, uint64_t value) {
    // MurmurHash64-inspired mixing
    constexpr uint64_t m = 0xc6a4a7935bd1e995ULL;
    constexpr int r = 47;

    value *= m;
    value ^= value >> r;
    value *= m;

    seed ^= value;
    seed *= m;
    seed += 0x9e3779b97f4a7c15ULL; // Golden ratio constant

    return seed;
}

__device__ uint64_t compute_genome_hash(const Genome* genome) {
    uint64_t hash = 0xcbf29ce484222325ULL; // FNV offset basis

    // Hash active segments only
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        if (genome->is_active(i)) {
            // Hash address tag
            hash = hash_combine(hash, genome->segments[i].address_tag);

            // Hash permission level and priority
            uint64_t meta = ((uint64_t)genome->segments[i].permission_level << 8) |
                           genome->segments[i].priority;
            hash = hash_combine(hash, meta);

            // Hash payload (sample every 8th float for speed)
            for (int j = 0; j < SEGMENT_PAYLOAD_SIZE; j += 8) {
                uint64_t val = __float_as_int(genome->segments[i].payload[j]);
                hash = hash_combine(hash, val);
            }
        }
    }

    return hash;
}

__global__ void compute_genome_hashes_kernel(Genome* genomes, uint64_t* hashes, int n) {
    int idx = GridStride::thread_id();
    if (idx < n) {
        hashes[idx] = compute_genome_hash(&genomes[idx]);
        genomes[idx].structural_hash = hashes[idx];
    }
}

// ============================================================================
// Mass Conservation Verification
// ============================================================================

__global__ void compute_total_mass_kernel(const float* concentration, float* result, int grid_cells, int channels) {
    __shared__ float sdata[BLOCK_SIZE_1D];

    float sum = 0.0f;
    for (int i = GridStride::thread_id(); i < grid_cells * channels; i += GridStride::stride()) {
        sum += concentration[i];
    }

    sum = BlockReduce<BLOCK_SIZE_1D>::sum(sum);

    if (threadIdx.x == 0) {
        Atomics::add_float(result, sum);
    }
}

// ============================================================================
// Global Reduction Kernels (leveraging tile_ops)
// ============================================================================

__global__ void reduce_max_fitness_kernel(const Organism* population, float* result, int n) {
    __shared__ float sdata[BLOCK_SIZE_1D / WARP_SIZE];

    float max_val = -1e10f;
    for (int i = GridStride::thread_id(); i < n; i += GridStride::stride()) {
        max_val = fmaxf(max_val, population[i].fitness);
    }

    // Warp-level reduction
    max_val = WarpReduce<WARP_SIZE>::max(max_val);

    // Write warp results to shared memory
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    if (lane == 0) {
        sdata[wid] = max_val;
    }
    __syncthreads();

    // Final reduction across warps
    if (wid == 0) {
        max_val = (lane < (BLOCK_SIZE_1D / WARP_SIZE)) ? sdata[lane] : -1e10f;
        max_val = WarpReduce<WARP_SIZE>::max(max_val);

        if (lane == 0) {
            atomicMax((int*)result, __float_as_int(max_val));
        }
    }
}

__global__ void reduce_sum_kernel(const float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE_1D];

    float sum = 0.0f;
    for (int i = GridStride::thread_id(); i < n; i += GridStride::stride()) {
        sum += input[i];
    }

    sum = BlockReduce<BLOCK_SIZE_1D>::sum(sum);

    if (threadIdx.x == 0) {
        Atomics::add_float(output, sum);
    }
}

// ============================================================================
// Variance and Covariance Kernels
// ============================================================================

__global__ void compute_variance_kernel(
    const float* values,
    float mean,
    float* variance,
    int n
) {
    __shared__ float sdata[BLOCK_SIZE_1D];

    float sum_sq_diff = 0.0f;
    for (int i = GridStride::thread_id(); i < n; i += GridStride::stride()) {
        float diff = values[i] - mean;
        sum_sq_diff += diff * diff;
    }

    sum_sq_diff = BlockReduce<BLOCK_SIZE_1D>::sum(sum_sq_diff);

    if (threadIdx.x == 0) {
        Atomics::add_float(variance, sum_sq_diff);
    }
}

// ============================================================================
// Atomic Min/Max for Floats (using tile_ops Atomics)
// ============================================================================

__device__ __forceinline__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int expected;

    do {
        expected = old;
        old = atomicCAS(address_as_int,
                       expected,
                       __float_as_int(fmaxf(val, __int_as_float(expected))));
    } while (expected != old);

    return __int_as_float(old);
}

__device__ __forceinline__ float atomicMinFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int expected;

    do {
        expected = old;
        old = atomicCAS(address_as_int,
                       expected,
                       __float_as_int(fminf(val, __int_as_float(expected))));
    } while (expected != old);

    return __int_as_float(old);
}

// ============================================================================
// Memory Initialization
// ============================================================================

__global__ void zero_memory_kernel(float* ptr, int n) {
    for (int i = GridStride::thread_id(); i < n; i += GridStride::stride()) {
        ptr[i] = 0.0f;
    }
}

__global__ void fill_memory_kernel(float* ptr, float value, int n) {
    for (int i = GridStride::thread_id(); i < n; i += GridStride::stride()) {
        ptr[i] = value;
    }
}

// ============================================================================
// Prefix Sum (Scan) for Compaction
// ============================================================================

__global__ void prefix_sum_kernel(const int* input, int* output, int n) {
    extern __shared__ int temp[];

    int tid = threadIdx.x;
    int offset = 1;

    // Load input into shared memory
    int idx = blockIdx.x * blockDim.x + tid;
    temp[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();

    // Build sum in place up the tree
    for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
        __syncthreads();
    }

    // Clear last element
    if (tid == 0) {
        temp[blockDim.x - 1] = 0;
    }
    __syncthreads();

    // Traverse down tree and build scan
    for (int d = 1; d < blockDim.x; d *= 2) {
        offset >>= 1;
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
        __syncthreads();
    }

    // Write results
    if (idx < n) {
        output[idx] = temp[tid];
    }
}

// ============================================================================
// Distance Computations (using FastMath from tile_ops)
// ============================================================================

__device__ __forceinline__ float behavioral_distance_sq(
    const float* a,
    const float* b
) {
    float dist_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < BEHAVIORAL_DIMS; i++) {
        float diff = a[i] - b[i];
        dist_sq += diff * diff;
    }
    return dist_sq;
}

__global__ void compute_pairwise_distances_kernel(
    const float* coords_a,
    const float* coords_b,
    float* distances,
    int n_a,
    int n_b
) {
    int idx = GridStride::thread_id();
    if (idx < n_a * n_b) {
        int i = idx / n_b;
        int j = idx % n_b;

        float dist_sq = behavioral_distance_sq(
            &coords_a[i * BEHAVIORAL_DIMS],
            &coords_b[j * BEHAVIORAL_DIMS]
        );
        distances[idx] = sqrtf(dist_sq);
    }
}

// ============================================================================
// Clamping and Parameter Mapping (using genome_to_param from tile_ops)
// ============================================================================

__device__ __forceinline__ float map_genome_param(float genome_val, float min_val, float max_val) {
    return genome_to_param(genome_val, min_val, max_val);
}

__global__ void expand_genome_params_kernel(
    const float* genome_vals,
    float* expanded_params,
    float min_val,
    float max_val,
    int n
) {
    int idx = GridStride::thread_id();
    if (idx < n) {
        expanded_params[idx] = genome_to_param(genome_vals[idx], min_val, max_val);
    }
}
