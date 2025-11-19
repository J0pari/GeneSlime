#include "../config/constants.cuh"
#include "../core/types.cuh"
#include "../utils/tile_ops.cuh"
#include "../kernels/utils.cu"
#include <cuda_runtime.h>

// Forward genome compression via structural analysis and deduplication
// Complements SVD compression in archive.cu with lossless structural compression

struct CompressionDictionary {
    float* common_patterns;     // [MAX_PATTERNS × PATTERN_SIZE]
    int* pattern_frequencies;   // [MAX_PATTERNS]
    uint32_t* pattern_hashes;   // [MAX_PATTERNS]
    int num_patterns;
    int pattern_size;
};

struct CompressedGenomeStructural {
    uint16_t* segment_pattern_ids;  // [NUM_SEGMENTS] indices into dictionary
    float* segment_residuals;       // [NUM_SEGMENTS × SEGMENT_PAYLOAD_SIZE] differences from pattern
    uint8_t* metadata;              // Permission, address, flags
    uint32_t compression_ratio;     // Encoded as fixed-point (1000 = 1.0x)
};

// Find nearest pattern in dictionary using L2 distance
__device__ int find_nearest_pattern(
    const float* payload,
    const CompressionDictionary* dict
) {
    float min_dist_sq = FLT_MAX;
    int nearest_idx = -1;

    for (int i = 0; i < dict->num_patterns; i++) {
        float dist_sq = 0.0f;

        #pragma unroll 8
        for (int j = 0; j < dict->pattern_size; j++) {
            float delta = payload[j] - dict->common_patterns[i * dict->pattern_size + j];
            dist_sq += delta * delta;
        }

        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            nearest_idx = i;
        }
    }

    return nearest_idx;
}

// Build compression dictionary from population via k-means clustering
__global__ void build_dictionary_kernel(
    const Genome* population,
    int population_size,
    CompressionDictionary* dict,
    int num_iterations,
    curandState* rand_states
) {
    using GridStride = GridStride1D;
    curandState local_state = rand_states[blockIdx.x * blockDim.x + threadIdx.x];

    // Initialize centroids randomly from population segments
    for (int k = GridStride::start(); k < dict->num_patterns; k += GridStride::stride()) {
        int random_genome = curand(&local_state) % population_size;
        int random_segment = curand(&local_state) % NUM_SEGMENTS;

        for (int i = 0; i < dict->pattern_size; i++) {
            dict->common_patterns[k * dict->pattern_size + i] =
                population[random_genome].segments[random_segment].payload[i];
        }
        dict->pattern_frequencies[k] = 0;
    }
    __syncthreads();

    // K-means iterations
    for (int iter = 0; iter < num_iterations; iter++) {
        __shared__ float new_centroids[MAX_PATTERNS * SEGMENT_PAYLOAD_SIZE];
        __shared__ int counts[MAX_PATTERNS];

        // Reset accumulators
        for (int k = threadIdx.x; k < dict->num_patterns; k += blockDim.x) {
            counts[k] = 0;
            for (int i = 0; i < dict->pattern_size; i++) {
                new_centroids[k * dict->pattern_size + i] = 0.0f;
            }
        }
        __syncthreads();

        // Assign each segment to nearest centroid
        for (int idx = GridStride::start(); idx < population_size * NUM_SEGMENTS; idx += GridStride::stride()) {
            int genome_idx = idx / NUM_SEGMENTS;
            int seg_idx = idx % NUM_SEGMENTS;

            const float* payload = population[genome_idx].segments[seg_idx].payload;
            int nearest = find_nearest_pattern(payload, dict);

            atomicAdd(&counts[nearest], 1);
            for (int i = 0; i < dict->pattern_size; i++) {
                atomicAdd(&new_centroids[nearest * dict->pattern_size + i], payload[i]);
            }
        }
        __syncthreads();

        // Update centroids
        for (int k = GridStride::start(); k < dict->num_patterns; k += GridStride::stride()) {
            if (counts[k] > 0) {
                for (int i = 0; i < dict->pattern_size; i++) {
                    dict->common_patterns[k * dict->pattern_size + i] =
                        new_centroids[k * dict->pattern_size + i] / counts[k];
                }
            }
        }
        __syncthreads();
    }

    // Compute pattern hashes and final frequencies
    for (int k = GridStride::start(); k < dict->num_patterns; k += GridStride::stride()) {
        // Use hash_combine from utils.cu to build pattern hash
        uint64_t hash = 0xcbf29ce484222325ULL;
        for (int i = 0; i < dict->pattern_size; i++) {
            hash = hash_combine(hash, __float_as_int(dict->common_patterns[k * dict->pattern_size + i]));
        }
        dict->pattern_hashes[k] = (uint32_t)(hash & 0xFFFFFFFF);

        int frequency = 0;
        for (int idx = 0; idx < population_size * NUM_SEGMENTS; idx++) {
            int genome_idx = idx / NUM_SEGMENTS;
            int seg_idx = idx % NUM_SEGMENTS;

            int nearest = find_nearest_pattern(population[genome_idx].segments[seg_idx].payload, dict);
            if (nearest == k) {
                frequency++;
            }
        }
        dict->pattern_frequencies[k] = frequency;
    }

    rand_states[blockIdx.x * blockDim.x + threadIdx.x] = local_state;
}

// Compress genome using dictionary
__global__ void compress_genome_forward_kernel(
    const Genome* source,
    CompressedGenomeStructural* compressed,
    const CompressionDictionary* dict
) {
    using GridStride = GridStride1D;

    int original_size = NUM_SEGMENTS * SEGMENT_PAYLOAD_SIZE * sizeof(float);
    int compressed_size = 0;

    for (int seg = GridStride::start(); seg < NUM_SEGMENTS; seg += GridStride::stride()) {
        const float* payload = source->segments[seg].payload;

        // Find best matching pattern
        int pattern_idx = find_nearest_pattern(payload, dict);
        compressed->segment_pattern_ids[seg] = pattern_idx;

        // Store residual (difference from pattern)
        for (int i = 0; i < SEGMENT_PAYLOAD_SIZE; i++) {
            float residual = payload[i] - dict->common_patterns[pattern_idx * dict->pattern_size + i];
            compressed->segment_residuals[seg * SEGMENT_PAYLOAD_SIZE + i] = residual;
        }

        // Store metadata (lossless)
        int meta_offset = seg * 4;
        compressed->metadata[meta_offset + 0] = source->segments[seg].permission_level;
        compressed->metadata[meta_offset + 1] = source->segments[seg].priority;
        compressed->metadata[meta_offset + 2] = source->segments[seg].mobility_flag ? 1 : 0;
        compressed->metadata[meta_offset + 3] = (uint8_t)(source->segments[seg].address_tag >> 8);
    }

    // Estimate compression ratio (pattern_id + sparse residual + metadata)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int pattern_ids_size = NUM_SEGMENTS * sizeof(uint16_t);
        int metadata_size = NUM_SEGMENTS * 4;

        // Count significant residuals (|residual| > threshold)
        int significant_residuals = 0;
        for (int seg = 0; seg < NUM_SEGMENTS; seg++) {
            for (int i = 0; i < SEGMENT_PAYLOAD_SIZE; i++) {
                if (fabsf(compressed->segment_residuals[seg * SEGMENT_PAYLOAD_SIZE + i]) > 1e-4f) {
                    significant_residuals++;
                }
            }
        }
        int residuals_size = significant_residuals * sizeof(float) + NUM_SEGMENTS * sizeof(int);  // Values + indices

        compressed_size = pattern_ids_size + metadata_size + residuals_size;
        compressed->compression_ratio = (compressed_size * 1000) / original_size;
    }
}

// Decompress genome from structural representation
__global__ void decompress_genome_forward_kernel(
    const CompressedGenomeStructural* compressed,
    const CompressionDictionary* dict,
    Genome* target
) {
    using GridStride = GridStride1D;

    for (int seg = GridStride::start(); seg < NUM_SEGMENTS; seg += GridStride::stride()) {
        int pattern_idx = compressed->segment_pattern_ids[seg];

        // Reconstruct payload = pattern + residual
        for (int i = 0; i < SEGMENT_PAYLOAD_SIZE; i++) {
            target->segments[seg].payload[i] =
                dict->common_patterns[pattern_idx * dict->pattern_size + i] +
                compressed->segment_residuals[seg * SEGMENT_PAYLOAD_SIZE + i];
        }

        // Restore metadata
        int meta_offset = seg * 4;
        target->segments[seg].permission_level = compressed->metadata[meta_offset + 0];
        target->segments[seg].priority = compressed->metadata[meta_offset + 1];
        target->segments[seg].mobility_flag = (compressed->metadata[meta_offset + 2] != 0);
        target->segments[seg].address_tag = ((uint16_t)compressed->metadata[meta_offset + 3]) << 8;
    }
}

// Delta compression for temporal sequences (parent → offspring)
__global__ void compress_delta_kernel(
    const Genome* parent,
    const Genome* offspring,
    CompressedGenomeStructural* delta_compressed
) {
    using GridStride = GridStride1D;

    for (int seg = GridStride::start(); seg < NUM_SEGMENTS; seg += GridStride::stride()) {
        // Store index of unchanged segments as 0xFFFF
        bool identical = true;

        for (int i = 0; i < SEGMENT_PAYLOAD_SIZE; i++) {
            if (parent->segments[seg].payload[i] != offspring->segments[seg].payload[i]) {
                identical = false;
                break;
            }
        }

        if (identical &&
            parent->segments[seg].permission_level == offspring->segments[seg].permission_level &&
            parent->segments[seg].address_tag == offspring->segments[seg].address_tag) {
            delta_compressed->segment_pattern_ids[seg] = 0xFFFF;
        } else {
            delta_compressed->segment_pattern_ids[seg] = seg;  // Mark as changed

            // Store delta
            for (int i = 0; i < SEGMENT_PAYLOAD_SIZE; i++) {
                delta_compressed->segment_residuals[seg * SEGMENT_PAYLOAD_SIZE + i] =
                    offspring->segments[seg].payload[i] - parent->segments[seg].payload[i];
            }

            int meta_offset = seg * 4;
            delta_compressed->metadata[meta_offset + 0] = offspring->segments[seg].permission_level;
            delta_compressed->metadata[meta_offset + 1] = offspring->segments[seg].priority;
            delta_compressed->metadata[meta_offset + 2] = offspring->segments[seg].mobility_flag ? 1 : 0;
            delta_compressed->metadata[meta_offset + 3] = (uint8_t)(offspring->segments[seg].address_tag >> 8);
        }
    }
}

// Decompress delta representation
__global__ void decompress_delta_kernel(
    const Genome* parent,
    const CompressedGenomeStructural* delta_compressed,
    Genome* offspring
) {
    using GridStride = GridStride1D;

    for (int seg = GridStride::start(); seg < NUM_SEGMENTS; seg += GridStride::stride()) {
        if (delta_compressed->segment_pattern_ids[seg] == 0xFFFF) {
            // Unchanged segment - copy from parent
            for (int i = 0; i < SEGMENT_PAYLOAD_SIZE; i++) {
                offspring->segments[seg].payload[i] = parent->segments[seg].payload[i];
            }
            offspring->segments[seg].permission_level = parent->segments[seg].permission_level;
            offspring->segments[seg].address_tag = parent->segments[seg].address_tag;
            offspring->segments[seg].priority = parent->segments[seg].priority;
            offspring->segments[seg].mobility_flag = parent->segments[seg].mobility_flag;
        } else {
            // Changed segment - reconstruct from delta
            for (int i = 0; i < SEGMENT_PAYLOAD_SIZE; i++) {
                offspring->segments[seg].payload[i] =
                    parent->segments[seg].payload[i] +
                    delta_compressed->segment_residuals[seg * SEGMENT_PAYLOAD_SIZE + i];
            }

            int meta_offset = seg * 4;
            offspring->segments[seg].permission_level = delta_compressed->metadata[meta_offset + 0];
            offspring->segments[seg].priority = delta_compressed->metadata[meta_offset + 1];
            offspring->segments[seg].mobility_flag = (delta_compressed->metadata[meta_offset + 2] != 0);
            offspring->segments[seg].address_tag = ((uint16_t)delta_compressed->metadata[meta_offset + 3]) << 8;
        }
    }
}

// Adaptive quantization for residuals
__global__ void quantize_residuals_kernel(
    CompressedGenomeStructural* compressed,
    int bits_per_value
) {
    using GridStride = GridStride1D;

    // Find min/max residual for normalization
    __shared__ float min_residual;
    __shared__ float max_residual;

    if (threadIdx.x == 0) {
        min_residual = FLT_MAX;
        max_residual = -FLT_MAX;
    }
    __syncthreads();

    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;

    for (int idx = GridStride::start(); idx < NUM_SEGMENTS * SEGMENT_PAYLOAD_SIZE; idx += GridStride::stride()) {
        float val = compressed->segment_residuals[idx];
        local_min = fminf(local_min, val);
        local_max = fmaxf(local_max, val);
    }

    using WarpReduce = WarpReduce<float>;
    local_min = WarpReduce::min(local_min);
    local_max = WarpReduce::max(local_max);

    if (threadIdx.x % WARP_SIZE == 0) {
        atomicMinFloat(&min_residual, local_min);  // From kernels/utils.cu
        atomicMaxFloat(&max_residual, local_max);  // From kernels/utils.cu
    }
    __syncthreads();

    // Quantize residuals to fixed-point representation
    int num_levels = (1 << bits_per_value) - 1;
    float range = max_residual - min_residual;

    for (int idx = GridStride::start(); idx < NUM_SEGMENTS * SEGMENT_PAYLOAD_SIZE; idx += GridStride::stride()) {
        float val = compressed->segment_residuals[idx];
        float normalized = (val - min_residual) / (range + 1e-10f);
        int quantized = __float2int_rn(normalized * num_levels);
        float dequantized = ((float)quantized / num_levels) * range + min_residual;
        compressed->segment_residuals[idx] = dequantized;
    }
}

// Compute compression statistics
__global__ void compute_compression_stats_kernel(
    const CompressedGenomeStructural* compressed_population,
    int population_size,
    float* avg_compression_ratio,
    float* avg_reconstruction_error
) {
    using BlockReduce = BlockReduce<float>;

    __shared__ float sum_ratio;
    __shared__ float sum_error;

    if (threadIdx.x == 0) {
        sum_ratio = 0.0f;
        sum_error = 0.0f;
    }
    __syncthreads();

    float local_ratio = 0.0f;
    float local_error = 0.0f;

    for (int i = threadIdx.x; i < population_size; i += blockDim.x) {
        local_ratio += compressed_population[i].compression_ratio / 1000.0f;

        // Estimate reconstruction error from residual magnitudes
        float error = 0.0f;
        for (int seg = 0; seg < NUM_SEGMENTS; seg++) {
            for (int j = 0; j < SEGMENT_PAYLOAD_SIZE; j++) {
                float residual = compressed_population[i].segment_residuals[seg * SEGMENT_PAYLOAD_SIZE + j];
                error += residual * residual;
            }
        }
        local_error += sqrtf(error / (NUM_SEGMENTS * SEGMENT_PAYLOAD_SIZE));
    }

    local_ratio = BlockReduce::sum(local_ratio);
    local_error = BlockReduce::sum(local_error);

    if (threadIdx.x == 0) {
        atomicAdd(&sum_ratio, local_ratio);
        atomicAdd(&sum_error, local_error);
    }
    __syncthreads();

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *avg_compression_ratio = sum_ratio / population_size;
        *avg_reconstruction_error = sum_error / population_size;
    }
}
