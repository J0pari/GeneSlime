#include "../config/constants.cuh"
#include "../core/types.cuh"
#include "../utils/tile_ops.cuh"
#include <cuda_runtime.h>

// Execution trace encoding for reverse synthesis
// Captures causal structure of genome → phenotype → behavior pathway
// Enables gradient-free credit assignment via trace reconstruction

struct ExecutionTrace {
    // Spatial activation maps per segment
    float* segment_activations;  // [NUM_SEGMENTS × GRID_SIZE × GRID_SIZE]

    // Temporal dynamics
    float* concentration_timeline;  // [NUM_TIMESTEPS × GRID_SIZE × GRID_SIZE × CHANNELS]
    float* flow_field_timeline;     // [NUM_TIMESTEPS × GRID_SIZE × GRID_SIZE × CHANNELS × 2]

    // Stigmergic write events
    uint32_t* write_events;         // [MAX_WRITE_EVENTS × 4] (segment_id, x, y, layer)
    float* write_magnitudes;        // [MAX_WRITE_EVENTS]
    int write_event_count;

    // Kernel contributions
    float* kernel_contributions;    // [NUM_HEADS × GRID_SIZE × GRID_SIZE]

    // Temporal correlation tracking
    float* segment_activation_history;  // [CORRELATION_WINDOW × NUM_SEGMENTS]
    float* prediction_error_history;    // [CORRELATION_WINDOW]
    float* fitness_history;             // [CORRELATION_WINDOW]
    float segment_correlation_matrix[NUM_SEGMENTS * NUM_SEGMENTS];
    int history_write_idx;

    // Behavioral endpoint
    float behavioral_coordinates[BEHAVIOR_DIM];
    float fitness;
};

struct TraceEncoder {
    ExecutionTrace* traces;
    int num_traces;
    int current_timestep;
    bool recording_enabled;
};

// Initialize trace encoder
__global__ void init_trace_encoder_kernel(
    TraceEncoder* encoder,
    int num_organisms,
    int max_timesteps
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    encoder->num_traces = num_organisms;
    encoder->current_timestep = 0;
    encoder->recording_enabled = true;

    for (int i = 0; i < num_organisms; i++) {
        encoder->traces[i].write_event_count = 0;
    }
}

// Record segment activation pattern
__global__ void record_segment_activation_kernel(
    TraceEncoder* encoder,
    int organism_id,
    int segment_id,
    const float* ca_weights,  // [GRID_SIZE × GRID_SIZE × weight_size]
    int weight_size
) {
    if (!encoder->recording_enabled) return;

    using GridStride = GridStride1D;
    ExecutionTrace* trace = &encoder->traces[organism_id];

    for (int idx = GridStride::start(); idx < GRID_SIZE * GRID_SIZE; idx += GridStride::stride()) {
        int y = idx / GRID_SIZE;
        int x = idx % GRID_SIZE;
        int cell_idx = y * GRID_SIZE + x;

        // Compute activation magnitude (L2 norm of weights contributed by this segment)
        float activation = 0.0f;
        for (int w = 0; w < weight_size; w++) {
            float val = ca_weights[cell_idx * weight_size + w];
            activation += val * val;
        }
        activation = sqrtf(activation);

        trace->segment_activations[segment_id * GRID_SIZE * GRID_SIZE + cell_idx] = activation;
    }
}

// Record CA concentration state at current timestep
__global__ void record_concentration_snapshot_kernel(
    TraceEncoder* encoder,
    int organism_id,
    const float* ca_concentration,
    int timestep
) {
    if (!encoder->recording_enabled) return;

    using GridStride = GridStride1D;
    ExecutionTrace* trace = &encoder->traces[organism_id];

    for (int idx = GridStride::start(); idx < GRID_SIZE * GRID_SIZE * CHANNELS; idx += GridStride::stride()) {
        int offset = timestep * GRID_SIZE * GRID_SIZE * CHANNELS + idx;
        trace->concentration_timeline[offset] = ca_concentration[idx];
    }
}

// Record flow field at current timestep
__global__ void record_flow_field_snapshot_kernel(
    TraceEncoder* encoder,
    int organism_id,
    const float* flow_field_x,
    const float* flow_field_y,
    int timestep
) {
    if (!encoder->recording_enabled) return;

    using GridStride = GridStride1D;
    ExecutionTrace* trace = &encoder->traces[organism_id];

    for (int idx = GridStride::start(); idx < GRID_SIZE * GRID_SIZE * CHANNELS; idx += GridStride::stride()) {
        int base_offset = timestep * GRID_SIZE * GRID_SIZE * CHANNELS * 2 + idx * 2;
        trace->flow_field_timeline[base_offset + 0] = flow_field_x[idx];
        trace->flow_field_timeline[base_offset + 1] = flow_field_y[idx];
    }
}

// Record stigmergic write event
__device__ void record_write_event(
    ExecutionTrace* trace,
    int segment_id,
    int grid_x,
    int grid_y,
    int layer,
    float magnitude
) {
    int event_idx = atomicAdd(&trace->write_event_count, 1);

    if (event_idx < MAX_WRITE_EVENTS) {
        trace->write_events[event_idx * 4 + 0] = segment_id;
        trace->write_events[event_idx * 4 + 1] = grid_x;
        trace->write_events[event_idx * 4 + 2] = grid_y;
        trace->write_events[event_idx * 4 + 3] = layer;
        trace->write_magnitudes[event_idx] = magnitude;
    }
}

// Record kernel contribution map for a specific head
__global__ void record_kernel_contribution_kernel(
    TraceEncoder* encoder,
    int organism_id,
    int head_idx,
    const float* affinity_map,
    const float* prev_affinity_map
) {
    if (!encoder->recording_enabled) return;

    using GridStride = GridStride1D;
    ExecutionTrace* trace = &encoder->traces[organism_id];

    for (int idx = GridStride::start(); idx < GRID_SIZE * GRID_SIZE; idx += GridStride::stride()) {
        // Compute contribution as difference from previous state
        float contribution = 0.0f;
        for (int c = 0; c < CHANNELS; c++) {
            float delta = affinity_map[idx * CHANNELS + c] - prev_affinity_map[idx * CHANNELS + c];
            contribution += fabsf(delta);
        }

        trace->kernel_contributions[head_idx * GRID_SIZE * GRID_SIZE + idx] = contribution;
    }
}

// Reverse synthesis: Identify most influential segments for target behavior
__global__ void reverse_synthesize_causal_segments_kernel(
    const ExecutionTrace* trace,
    const float* target_behavior,
    float* segment_causal_scores,  // Output: [NUM_SEGMENTS]
    float behavior_tolerance
) {
    using GridStride = GridStride1D;

    __shared__ float shared_scores[NUM_SEGMENTS];
    if (threadIdx.x < NUM_SEGMENTS) {
        shared_scores[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Compute behavioral distance
    float behavior_dist_sq = 0.0f;
    for (int d = 0; d < BEHAVIOR_DIM; d++) {
        float delta = trace->behavioral_coordinates[d] - target_behavior[d];
        behavior_dist_sq += delta * delta;
    }
    float behavior_dist = sqrtf(behavior_dist_sq);

    // If trace is not close to target, skip
    if (behavior_dist > behavior_tolerance) {
        if (threadIdx.x < NUM_SEGMENTS) {
            segment_causal_scores[threadIdx.x] = 0.0f;
        }
        return;
    }

    // Trace backwards: Score segments by their contribution to successful behavior
    for (int seg = 0; seg < NUM_SEGMENTS; seg++) {
        float causal_score = 0.0f;

        // 1. Direct spatial activation
        for (int idx = GridStride::start(); idx < GRID_SIZE * GRID_SIZE; idx += GridStride::stride()) {
            causal_score += trace->segment_activations[seg * GRID_SIZE * GRID_SIZE + idx];
        }

        // 2. Stigmergic write influence
        for (int event_idx = 0; event_idx < trace->write_event_count; event_idx++) {
            if (trace->write_events[event_idx * 4 + 0] == seg) {
                causal_score += trace->write_magnitudes[event_idx] * 10.0f;  // Weight stigmergic influence
            }
        }

        // Aggregate across thread block
        using WarpReduce = WarpReduce<float>;
        float warp_score = WarpReduce::sum(causal_score);
        if (threadIdx.x % WARP_SIZE == 0) {
            atomicAdd(&shared_scores[seg], warp_score);
        }
    }

    __syncthreads();

    // Write results
    if (threadIdx.x < NUM_SEGMENTS) {
        segment_causal_scores[threadIdx.x] = shared_scores[threadIdx.x];
    }
}

// Compute temporal derivative of concentration (rate of change)
__global__ void compute_concentration_derivative_kernel(
    const ExecutionTrace* trace,
    float* temporal_derivative,  // Output: [NUM_TIMESTEPS-1 × GRID_SIZE × GRID_SIZE × CHANNELS]
    int num_timesteps
) {
    using GridStride = GridStride1D;

    for (int idx = GridStride::start(); idx < (num_timesteps - 1) * GRID_SIZE * GRID_SIZE * CHANNELS; idx += GridStride::stride()) {
        int t = idx / (GRID_SIZE * GRID_SIZE * CHANNELS);
        int spatial_channel_idx = idx % (GRID_SIZE * GRID_SIZE * CHANNELS);

        float current = trace->concentration_timeline[t * GRID_SIZE * GRID_SIZE * CHANNELS + spatial_channel_idx];
        float next = trace->concentration_timeline[(t + 1) * GRID_SIZE * GRID_SIZE * CHANNELS + spatial_channel_idx];

        temporal_derivative[idx] = next - current;
    }
}

// Identify critical timesteps (moments of high rate of change)
__global__ void identify_critical_timesteps_kernel(
    const float* temporal_derivative,
    int* critical_timesteps,  // Output: sorted indices
    int* num_critical,
    int num_timesteps,
    float threshold_percentile
) {
    using BlockReduce = BlockReduce<float>;

    __shared__ float derivative_magnitudes[MAX_TIMESTEPS];
    __shared__ int sorted_indices[MAX_TIMESTEPS];

    // Compute magnitude of change at each timestep
    for (int t = threadIdx.x; t < num_timesteps - 1; t += blockDim.x) {
        float magnitude = 0.0f;
        for (int idx = 0; idx < GRID_SIZE * GRID_SIZE * CHANNELS; idx++) {
            float deriv = temporal_derivative[t * GRID_SIZE * GRID_SIZE * CHANNELS + idx];
            magnitude += deriv * deriv;
        }
        derivative_magnitudes[t] = sqrtf(magnitude);
        sorted_indices[t] = t;
    }
    __syncthreads();

    // Bubble sort to identify top percentile (simple but sufficient for small num_timesteps)
    if (threadIdx.x == 0) {
        for (int i = 0; i < num_timesteps - 2; i++) {
            for (int j = 0; j < num_timesteps - 2 - i; j++) {
                if (derivative_magnitudes[sorted_indices[j]] < derivative_magnitudes[sorted_indices[j + 1]]) {
                    int temp = sorted_indices[j];
                    sorted_indices[j] = sorted_indices[j + 1];
                    sorted_indices[j + 1] = temp;
                }
            }
        }

        int num_keep = (int)((num_timesteps - 1) * threshold_percentile);
        *num_critical = num_keep;
        for (int i = 0; i < num_keep; i++) {
            critical_timesteps[i] = sorted_indices[i];
        }
    }
}

// Compute segment-to-behavior influence matrix via trace analysis
__global__ void compute_influence_matrix_kernel(
    const ExecutionTrace* traces,
    int num_traces,
    float* influence_matrix,  // Output: [NUM_SEGMENTS × BEHAVIOR_DIM]
    float regularization
) {
    using GridStride = GridStride1D;

    // For each segment-behavior pair, compute covariance across traces
    for (int idx = GridStride::start(); idx < NUM_SEGMENTS * BEHAVIOR_DIM; idx += GridStride::stride()) {
        int seg = idx / BEHAVIOR_DIM;
        int b_dim = idx % BEHAVIOR_DIM;

        float seg_mean = 0.0f;
        float behavior_mean = 0.0f;

        // Compute means
        for (int t = 0; t < num_traces; t++) {
            float seg_activation = 0.0f;
            for (int cell = 0; cell < GRID_SIZE * GRID_SIZE; cell++) {
                seg_activation += traces[t].segment_activations[seg * GRID_SIZE * GRID_SIZE + cell];
            }
            seg_activation /= (GRID_SIZE * GRID_SIZE);

            seg_mean += seg_activation;
            behavior_mean += traces[t].behavioral_coordinates[b_dim];
        }
        seg_mean /= num_traces;
        behavior_mean /= num_traces;

        // Compute covariance
        float covariance = 0.0f;
        float seg_variance = 0.0f;

        for (int t = 0; t < num_traces; t++) {
            float seg_activation = 0.0f;
            for (int cell = 0; cell < GRID_SIZE * GRID_SIZE; cell++) {
                seg_activation += traces[t].segment_activations[seg * GRID_SIZE * GRID_SIZE + cell];
            }
            seg_activation /= (GRID_SIZE * GRID_SIZE);

            float seg_dev = seg_activation - seg_mean;
            float behavior_dev = traces[t].behavioral_coordinates[b_dim] - behavior_mean;

            covariance += seg_dev * behavior_dev;
            seg_variance += seg_dev * seg_dev;
        }

        // Regularized influence coefficient
        float influence = covariance / (seg_variance + regularization);
        influence_matrix[idx] = influence;
    }
}

// Compress trace for long-term storage (keep only critical timesteps)
__global__ void compress_trace_kernel(
    const ExecutionTrace* full_trace,
    ExecutionTrace* compressed_trace,
    const int* critical_timesteps,
    int num_critical_timesteps
) {
    using GridStride = GridStride1D;

    // Copy only critical timesteps
    for (int t = 0; t < num_critical_timesteps; t++) {
        int source_t = critical_timesteps[t];

        for (int idx = GridStride::start(); idx < GRID_SIZE * GRID_SIZE * CHANNELS; idx += GridStride::stride()) {
            compressed_trace->concentration_timeline[t * GRID_SIZE * GRID_SIZE * CHANNELS + idx] =
                full_trace->concentration_timeline[source_t * GRID_SIZE * GRID_SIZE * CHANNELS + idx];

            int flow_idx = idx * 2;
            compressed_trace->flow_field_timeline[t * GRID_SIZE * GRID_SIZE * CHANNELS * 2 + flow_idx + 0] =
                full_trace->flow_field_timeline[source_t * GRID_SIZE * GRID_SIZE * CHANNELS * 2 + flow_idx + 0];
            compressed_trace->flow_field_timeline[t * GRID_SIZE * GRID_SIZE * CHANNELS * 2 + flow_idx + 1] =
                full_trace->flow_field_timeline[source_t * GRID_SIZE * GRID_SIZE * CHANNELS * 2 + flow_idx + 1];
        }
    }

    // Copy segment activations and write events (these are already compact)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int seg = 0; seg < NUM_SEGMENTS; seg++) {
            for (int cell = 0; cell < GRID_SIZE * GRID_SIZE; cell++) {
                compressed_trace->segment_activations[seg * GRID_SIZE * GRID_SIZE + cell] =
                    full_trace->segment_activations[seg * GRID_SIZE * GRID_SIZE + cell];
            }
        }

        for (int event = 0; event < full_trace->write_event_count; event++) {
            for (int i = 0; i < 4; i++) {
                compressed_trace->write_events[event * 4 + i] = full_trace->write_events[event * 4 + i];
            }
            compressed_trace->write_magnitudes[event] = full_trace->write_magnitudes[event];
        }
        compressed_trace->write_event_count = full_trace->write_event_count;
    }
}

// Update temporal correlation tracking (called each timestep during evaluation)
__global__ void update_correlation_tracking_kernel(
    TraceEncoder* encoder,
    int organism_id,
    const float* segment_activations,  // Current segment activation levels [NUM_SEGMENTS]
    float prediction_error,
    float fitness
) {
    if (!encoder->recording_enabled) return;

    ExecutionTrace* trace = &encoder->traces[organism_id];
    int write_idx = trace->history_write_idx % CORRELATION_WINDOW;

    // Update histories (circular buffer)
    if (threadIdx.x < NUM_SEGMENTS) {
        trace->segment_activation_history[write_idx * NUM_SEGMENTS + threadIdx.x] = segment_activations[threadIdx.x];
    }

    if (threadIdx.x == 0) {
        trace->prediction_error_history[write_idx] = prediction_error;
        trace->fitness_history[write_idx] = fitness;
        trace->history_write_idx++;
    }
}

// Compute segment correlation matrix from activation history
__global__ void compute_segment_correlation_matrix_kernel(
    ExecutionTrace* trace,
    int history_length
) {
    int seg_i = blockIdx.x;
    int seg_j = threadIdx.x;

    if (seg_i >= NUM_SEGMENTS || seg_j >= NUM_SEGMENTS) return;

    // Compute correlation between segment i and segment j across time
    float mean_i = 0.0f;
    float mean_j = 0.0f;

    for (int t = 0; t < history_length; t++) {
        mean_i += trace->segment_activation_history[t * NUM_SEGMENTS + seg_i];
        mean_j += trace->segment_activation_history[t * NUM_SEGMENTS + seg_j];
    }
    mean_i /= history_length;
    mean_j /= history_length;

    float cov = 0.0f;
    float var_i = 0.0f;
    float var_j = 0.0f;

    for (int t = 0; t < history_length; t++) {
        float dev_i = trace->segment_activation_history[t * NUM_SEGMENTS + seg_i] - mean_i;
        float dev_j = trace->segment_activation_history[t * NUM_SEGMENTS + seg_j] - mean_j;

        cov += dev_i * dev_j;
        var_i += dev_i * dev_i;
        var_j += dev_j * dev_j;
    }

    float correlation = cov / (sqrtf(var_i * var_j) + 1e-10f);
    trace->segment_correlation_matrix[seg_i * NUM_SEGMENTS + seg_j] = correlation;
}

// Clear trace for new evaluation
__global__ void reset_trace_kernel(
    TraceEncoder* encoder,
    int organism_id
) {
    using GridStride = GridStride1D;
    ExecutionTrace* trace = &encoder->traces[organism_id];

    for (int idx = GridStride::start(); idx < NUM_SEGMENTS * GRID_SIZE * GRID_SIZE; idx += GridStride::stride()) {
        trace->segment_activations[idx] = 0.0f;
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        trace->write_event_count = 0;
        trace->history_write_idx = 0;
        encoder->current_timestep = 0;
    }
}
