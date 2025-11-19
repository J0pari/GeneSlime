#ifndef PARALLEL_COMPACTION_CU
#define PARALLEL_COMPACTION_CU

#include "../config/constants.cuh"
#include "../core/types.cuh"
#include "../utils/tile_ops.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

constexpr float DEFAULT_DELTA_THRESHOLD = 0.01f;

__global__ void warp_compact_deltas_kernel(
    Genome* child_genome,
    Genome* parent_genome,
    uint16_t* delta_indices,
    float* delta_values,
    uint16_t* output_count,
    float threshold = DEFAULT_DELTA_THRESHOLD
) {
    __shared__ uint16_t warp_counts[BLOCK_SIZE_1D / WARP_SIZE];
    __shared__ uint16_t warp_offsets[BLOCK_SIZE_1D / WARP_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    float diff = 0.0f;
    int predicate = 0;

    if (gid < TOTAL_GENOME_WEIGHTS) {
        int seg = gid / SEGMENT_PAYLOAD_SIZE;
        int w = gid % SEGMENT_PAYLOAD_SIZE;
        float child_val = child_genome->segments[seg].payload[w];
        float parent_val = parent_genome->segments[seg].payload[w];
        diff = child_val - parent_val;
        predicate = (fabsf(diff) > threshold) ? 1 : 0;
    }

    unsigned mask = __activemask();
    unsigned ballot = __ballot_sync(mask, predicate);
    int warp_count = __popc(ballot);
    int prefix_sum = __popc(ballot & ((1u << lane_id) - 1));

    if (lane_id == 0) {
        warp_counts[warp_id] = warp_count;
    }
    __syncthreads();

    if (tid == 0) {
        uint16_t running_sum = 0;
        for (int i = 0; i < blockDim.x / WARP_SIZE; i++) {
            warp_offsets[i] = running_sum;
            running_sum += warp_counts[i];
        }
        warp_counts[0] = running_sum;
    }
    __syncthreads();

    uint16_t block_total = warp_counts[0];
    uint16_t block_offset = 0;

    if (tid == 0) {
        block_offset = atomicAdd((unsigned int*)output_count, block_total);
    }
    __syncthreads();

    block_offset = __shfl_sync(0xFFFFFFFF, block_offset, 0);

    if (predicate) {
        int output_idx = block_offset + warp_offsets[warp_id] + prefix_sum;
        delta_indices[output_idx] = gid;
        delta_values[output_idx] = diff;
    }
}

__global__ void apply_deltas_kernel(
    Genome* parent_genome,
    uint16_t* delta_indices,
    float* delta_values,
    uint16_t num_deltas,
    Genome* output_genome
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < NUM_SEGMENTS) {
        for (int w = 0; w < SEGMENT_PAYLOAD_SIZE; w++) {
            output_genome->segments[tid].payload[w] = parent_genome->segments[tid].payload[w];
        }
        output_genome->segments[tid].address_tag = parent_genome->segments[tid].address_tag;
        output_genome->segments[tid].permission_level = parent_genome->segments[tid].permission_level;
        output_genome->segments[tid].priority = parent_genome->segments[tid].priority;
        output_genome->segments[tid].mobility_flag = parent_genome->segments[tid].mobility_flag;
        output_genome->segments[tid].expression_frequency = parent_genome->segments[tid].expression_frequency;
        output_genome->segments[tid].causal_attribution = parent_genome->segments[tid].causal_attribution;
    }

    if (tid < num_deltas) {
        uint16_t idx = delta_indices[tid];
        if (idx < TOTAL_GENOME_WEIGHTS) {
            int seg = idx / SEGMENT_PAYLOAD_SIZE;
            int w = idx % SEGMENT_PAYLOAD_SIZE;
            atomicAdd(&output_genome->segments[seg].payload[w], delta_values[tid]);
        }
    }
}

__global__ void shrink_delta_arrays_kernel(
    uint16_t* src_indices,
    float* src_values,
    uint16_t* dst_indices,
    float* dst_values,
    uint16_t num_deltas
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_deltas) {
        dst_indices[tid] = src_indices[tid];
        dst_values[tid] = src_values[tid];
    }
}

__global__ void exclusive_scan_kernel(
    int* input,
    int* output,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    __shared__ int warp_sums[WARP_SIZE];

    int val = (tid < N) ? input[tid] : 0;

    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
        int n = warp.shfl_up(val, offset);
        if (lane >= offset) val += n;
    }

    if (lane == WARP_SIZE - 1) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        int warp_sum = (lane < (blockDim.x / WARP_SIZE)) ? warp_sums[lane] : 0;

        #pragma unroll
        for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
            int n = warp.shfl_up(warp_sum, offset);
            if (lane >= offset) warp_sum += n;
        }

        warp_sums[lane] = warp_sum;
    }
    __syncthreads();

    int warp_offset = (warp_id > 0) ? warp_sums[warp_id - 1] : 0;
    int exclusive_val = warp_offset + val - ((tid < N) ? input[tid] : 0);

    if (tid < N) {
        output[tid] = exclusive_val;
    }
}

#endif
