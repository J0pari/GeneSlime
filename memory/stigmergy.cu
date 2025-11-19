#include <cuda_runtime.h>
#include "../core/types.cuh"
#include "../utils/tile_ops.cuh"

// ============================================================================
// Threshold-Gated Write to Stigmergic Layers
// ============================================================================

__global__ void write_stigmergic_layers_kernel(
    StigmergicField* field,
    const float* ca_concentration_prev,  // A^t
    const float* ca_concentration_next,  // A^(t+dt)
    int grid_cells
) {
    int cell_idx = GridStride::thread_id();
    if (cell_idx >= grid_cells) return;

    // Compute morphological change magnitude
    float signal = 0.0f;
    for (int c = 0; c < CHANNELS; c++) {
        float delta = ca_concentration_next[cell_idx * CHANNELS + c] -
                     ca_concentration_prev[cell_idx * CHANNELS + c];
        signal += fabsf(delta);
    }

    // Fast layer: low threshold
    if (signal >= field->write_thresholds[0]) {
        float excess = signal - field->write_thresholds[0];
        atomicAdd(&field->fast_layer[cell_idx], field->write_gains[0] * excess);
    }

    // Medium layer: moderate threshold
    if (signal >= field->write_thresholds[1]) {
        float excess = signal - field->write_thresholds[1];
        atomicAdd(&field->medium_layer[cell_idx], field->write_gains[1] * excess);
    }

    // Slow layer: high threshold
    if (signal >= field->write_thresholds[2]) {
        float excess = signal - field->write_thresholds[2];
        atomicAdd(&field->slow_layer[cell_idx], field->write_gains[2] * excess);
    }

    // Structural layer: extreme events only
    if (signal >= field->write_thresholds[3]) {
        float excess = signal - field->write_thresholds[3];
        atomicAdd(&field->structural_layer[cell_idx], field->write_gains[3] * excess);
    }
}

// ============================================================================
// Decay Stigmergic Layers (Ratchet Constraint)
// ============================================================================

__global__ void decay_stigmergic_layers_kernel(
    StigmergicField* field,
    int grid_cells
) {
    int cell_idx = GridStride::thread_id();
    if (cell_idx >= grid_cells) return;

    // Exponential decay per layer
    field->fast_layer[cell_idx] *= field->decay_rates[0];    // 0.9 (fast decay)
    field->medium_layer[cell_idx] *= field->decay_rates[1];  // 0.99
    field->slow_layer[cell_idx] *= field->decay_rates[2];    // 0.999

    // Structural layer: NO DECAY (decay_rates[3] = 1.0)
    // This enforces the ratchet constraint: lower layers cannot erase higher layers
}

// ============================================================================
// Weighted Read from Stigmergic Layers
// ============================================================================

__global__ void read_stigmergic_composite_kernel(
    const StigmergicField* field,
    const float* genome_read_weights,  // [POPULATION_SIZE × NUM_STIGMERGY_LAYERS]
    float* composite_output,            // [POPULATION_SIZE × GRID_CELLS_2D]
    int pop_size,
    int grid_cells
) {
    int org_idx = blockIdx.x;
    int cell_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (org_idx >= pop_size || cell_idx >= grid_cells) return;

    const float* weights = &genome_read_weights[org_idx * NUM_STIGMERGY_LAYERS];

    float composite = weights[0] * field->fast_layer[cell_idx] +
                     weights[1] * field->medium_layer[cell_idx] +
                     weights[2] * field->slow_layer[cell_idx] +
                     weights[3] * field->structural_layer[cell_idx];

    composite_output[org_idx * grid_cells + cell_idx] = composite;
}

// ============================================================================
// Initialize Stigmergic Field
// ============================================================================

__global__ void initialize_stigmergic_field_kernel(
    StigmergicField* field,
    int grid_cells
) {
    int idx = GridStride::thread_id();
    if (idx < grid_cells) {
        field->fast_layer[idx] = 0.0f;
        field->medium_layer[idx] = 0.0f;
        field->slow_layer[idx] = 0.0f;
        field->structural_layer[idx] = 0.0f;
    }

    // Initialize parameters (thread 0 only)
    if (idx == 0) {
        field->decay_rates[0] = DECAY_FAST;
        field->decay_rates[1] = DECAY_MEDIUM;
        field->decay_rates[2] = DECAY_SLOW;
        field->decay_rates[3] = DECAY_STRUCTURAL;

        field->write_thresholds[0] = THRESHOLD_FAST;
        field->write_thresholds[1] = THRESHOLD_MEDIUM;
        field->write_thresholds[2] = THRESHOLD_SLOW;
        field->write_thresholds[3] = THRESHOLD_STRUCTURAL;

        field->write_gains[0] = WRITE_GAIN_FAST;
        field->write_gains[1] = WRITE_GAIN_MEDIUM;
        field->write_gains[2] = WRITE_GAIN_SLOW;
        field->write_gains[3] = WRITE_GAIN_STRUCTURAL;
    }
}

// ============================================================================
// Stigmergic Attention: Organism-Specific Read Weights
// ============================================================================

__global__ void compute_read_weights_from_genome_kernel(
    const Organism* population,
    float* read_weights,  // [POPULATION_SIZE × NUM_STIGMERGY_LAYERS]
    int pop_size
) {
    int idx = GridStride::thread_id();
    if (idx < pop_size) {
        const Organism* org = &population[idx];

        // Extract read weights from regulatory segments
        float weights[NUM_STIGMERGY_LAYERS] = {0.25f, 0.25f, 0.25f, 0.25f}; // Default: equal attention

        // Use regulatory segment payloads to modulate attention
        for (int seg_idx = 12; seg_idx < 16; seg_idx++) { // Regulatory segments
            if (org->genome.is_active(seg_idx)) {
                int layer_idx = seg_idx - 12;
                // Average first 4 payload values as attention weight
                float avg = 0.0f;
                for (int i = 0; i < 4; i++) {
                    avg += org->genome.segments[seg_idx].payload[i];
                }
                avg /= 4.0f;
                weights[layer_idx] = fmaxf(0.0f, avg); // Ensure non-negative
            }
        }

        // Normalize weights
        float total = weights[0] + weights[1] + weights[2] + weights[3] + 1e-6f;
        for (int i = 0; i < NUM_STIGMERGY_LAYERS; i++) {
            read_weights[idx * NUM_STIGMERGY_LAYERS + i] = weights[i] / total;
        }
    }
}

// ============================================================================
// Stigmergic Pattern Extraction (for visualization/analysis)
// ============================================================================

__global__ void extract_stigmergic_patterns_kernel(
    const StigmergicField* field,
    float* pattern_output,  // [NUM_STIGMERGY_LAYERS × GRID_CELLS_2D]
    int grid_cells
) {
    int cell_idx = GridStride::thread_id();
    if (cell_idx < grid_cells) {
        pattern_output[0 * grid_cells + cell_idx] = field->fast_layer[cell_idx];
        pattern_output[1 * grid_cells + cell_idx] = field->medium_layer[cell_idx];
        pattern_output[2 * grid_cells + cell_idx] = field->slow_layer[cell_idx];
        pattern_output[3 * grid_cells + cell_idx] = field->structural_layer[cell_idx];
    }
}
