#ifndef TYPES_CUH
#define TYPES_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include "../config/constants.cuh"

// Forward declarations for device-side allocations
struct Genome;
struct StigmergicField;
struct ArchiveElite;
struct PopulationMetrics;

// ============================================================================
// Genome Structures
// ============================================================================

struct SegmentVariant {
    float expansion[SEGMENT_PAYLOAD_SIZE];  // Alternative weight pattern
    float base_probability;                  // Default selection probability
    float stress_modifier;                   // Probability shift under low coherence
};

struct StochasticSegment {
    SegmentVariant variant_A;   // Conservative (70% baseline)
    SegmentVariant variant_B;   // Exploratory (25% baseline)
    SegmentVariant variant_C;   // Disruptive (5% baseline)
    float selection_logic[4];   // Coefficients for probability computation
};

struct GenomeSegment {
    float payload[SEGMENT_PAYLOAD_SIZE];    // Weight values (64 floats)
    uint16_t address_tag;                    // Target location encoding
    uint8_t permission_level;                // Write permission (0=protected, 3=regulatory)
    uint8_t priority;                        // Dominance in address conflicts
    bool mobility_flag;                      // Can transpose?
    float expression_frequency;              // How often expressed (runtime updated)
    float causal_attribution;                // Correlation with fitness events
    StochasticSegment stochastic;            // Variant expansion data

    // Address tag decoding helpers
    __device__ __forceinline__ int get_grid_x() const { return (address_tag >> 10) & 0x3F; }
    __device__ __forceinline__ int get_grid_y() const { return (address_tag >> 4) & 0x3F; }
    __device__ __forceinline__ int get_head_index() const { return (address_tag >> 1) & 0x7; }
    __device__ __forceinline__ int get_variant_select() const { return address_tag & 0x1; }

    __device__ __forceinline__ void set_address(int gx, int gy, int head, int variant) {
        address_tag = ((gx & 0x3F) << 10) | ((gy & 0x3F) << 4) | ((head & 0x7) << 1) | (variant & 0x1);
    }
};

struct Genome {
    GenomeSegment segments[NUM_SEGMENTS];    // 16 segments × 64 floats = 1024 total
    uint32_t active_mask;                    // Which segments currently active (bit flags)
    float assembly_bias[NUM_SEGMENTS];       // Per-segment stochastic assembly weights
    uint64_t structural_hash;                // For deduplication

    __device__ __forceinline__ bool is_active(int idx) const {
        return (active_mask & (1 << idx)) != 0;
    }

    __device__ __forceinline__ void set_active(int idx, bool active) {
        if (active) {
            active_mask |= (1 << idx);
        } else {
            active_mask &= ~(1 << idx);
        }
    }
};

// ============================================================================
// Multi-Timescale Stigmergic Substrate
// ============================================================================

struct StigmergicField {
    float* fast_layer;          // [GRID_CELLS_2D] τ=1 gen, recent activity
    float* medium_layer;        // [GRID_CELLS_2D] τ=10 gen, short-term memory
    float* slow_layer;          // [GRID_CELLS_2D] τ=100 gen, long-term history
    float* structural_layer;    // [GRID_CELLS_2D] τ=∞, permanent modifications

    float decay_rates[NUM_STIGMERGY_LAYERS];      // {0.9f, 0.99f, 0.999f, 1.0f}
    float write_thresholds[NUM_STIGMERGY_LAYERS]; // {0.01f, 0.1f, 0.5f, 2.0f}
    float write_gains[NUM_STIGMERGY_LAYERS];      // {0.3f, 0.2f, 0.1f, 0.05f}

    __device__ __forceinline__ float read_composite(int cell_idx, const float* genome_read_weights) const {
        return genome_read_weights[0] * fast_layer[cell_idx] +
               genome_read_weights[1] * medium_layer[cell_idx] +
               genome_read_weights[2] * slow_layer[cell_idx] +
               genome_read_weights[3] * structural_layer[cell_idx];
    }
};

// ============================================================================
// Flow-Lenia State
// ============================================================================

struct MultiHeadCAState {
    float* ca_concentration;        // [GRID_CELLS_2D × CHANNELS] A^t
    half* perception_weights;       // [NUM_HEADS × CHANNELS × HIDDEN_DIM]
    half* interaction_weights;      // [NUM_HEADS × HIDDEN_DIM × HIDDEN_DIM]
    half* value_weights;            // [NUM_HEADS × HIDDEN_DIM × HEAD_DIM]
    float* ca_output;               // [NUM_HEADS × GRID_CELLS_2D × HEAD_DIM]
    float* affinity_reduced;        // [GRID_CELLS_2D] U^t
    float* flow_field_x;            // [GRID_CELLS_2D] F^t_x
    float* flow_field_y;            // [GRID_CELLS_2D] F^t_y
    float* reintegration_buffer;    // [GRID_CELLS_2D × CHANNELS] A^(t+dt)

    // Mass conservation tracking
    float mass_prev;
    float mass_current;

    __device__ __forceinline__ float get_concentration(int x, int y, int channel) const {
        return ca_concentration[(y * GRID_SIZE + x) * CHANNELS + channel];
    }

    __device__ __forceinline__ void set_concentration(int x, int y, int channel, float val) {
        ca_concentration[(y * GRID_SIZE + x) * CHANNELS + channel] = val;
    }
};

// ============================================================================
// Archive and MAP-Elites
// ============================================================================

struct ArchiveElite {
    // Forward encoding (deterministic)
    GenomeSegment compressed_genome[NUM_SEGMENTS];
    uint64_t genome_hash;

    // Reverse trace summary (probabilistic)
    float execution_trace[NUM_SEGMENTS];        // Which segments expressed during morphology
    float activation_frequency[NUM_SEGMENTS];   // How often each segment influenced dynamics
    float causal_attribution[NUM_SEGMENTS];     // Correlation with fitness events

    // Phenotype summary
    float behavioral_coords[BEHAVIORAL_DIMS];   // DIRESA embedding
    float fitness;
    float coherence;
    float effective_rank;

    // Genealogy
    uint32_t parent_ids[2];
    uint32_t elite_id;
    uint32_t generation;

    __device__ __forceinline__ float distance_to(const ArchiveElite& other) const {
        float dist_sq = 0.0f;
        #pragma unroll
        for (int i = 0; i < BEHAVIORAL_DIMS; i++) {
            float diff = behavioral_coords[i] - other.behavioral_coords[i];
            dist_sq += diff * diff;
        }
        return sqrtf(dist_sq);
    }
};

struct VoronoiCell {
    float center[BEHAVIORAL_DIMS];  // Cell centroid in behavioral space
    int elite_count;                 // Number of elites in this cell
    float best_fitness;              // Best fitness in cell
    int best_elite_idx;              // Index of best elite

    __device__ __forceinline__ float distance_to_point(const float* point) const {
        float dist_sq = 0.0f;
        #pragma unroll
        for (int i = 0; i < BEHAVIORAL_DIMS; i++) {
            float diff = center[i] - point[i];
            dist_sq += diff * diff;
        }
        return sqrtf(dist_sq);
    }
};

struct Archive {
    ArchiveElite* elites;                       // [MAX_ARCHIVE_SIZE]
    VoronoiCell* voronoi_cells;                 // [NUM_VORONOI_CELLS]
    int* elite_count;                            // Current number of elites
    int* cell_assignments;                       // [MAX_ARCHIVE_SIZE] elite → cell mapping

    __device__ __forceinline__ int find_voronoi_cell(const float* behavioral_coords) const {
        int best_cell = 0;
        float best_dist = 1e10f;

        for (int i = 0; i < NUM_VORONOI_CELLS; i++) {
            float dist = voronoi_cells[i].distance_to_point(behavioral_coords);
            if (dist < best_dist) {
                best_dist = dist;
                best_cell = i;
            }
        }
        return best_cell;
    }
};

// ============================================================================
// Population Metrics (Global State Sensors)
// ============================================================================

struct PopulationMetrics {
    float archive_saturation[NUM_VORONOI_CELLS];  // Per Voronoi cell
    float fitness_variance;                        // Population evolvability
    float fitness_mean;
    float coherence_mean;                          // Learning health
    float coherence_variance;
    float behavioral_novelty_rate;                 // Discovery rate
    float effective_rank_mean;
    int generation;
    bool crisis_mode;                              // Lifecycle crisis detected

    __device__ __forceinline__ float get_mean_saturation() const {
        float sum = 0.0f;
        #pragma unroll 16
        for (int i = 0; i < NUM_VORONOI_CELLS; i++) {
            sum += archive_saturation[i];
        }
        return sum / NUM_VORONOI_CELLS;
    }
};

// ============================================================================
// Organism State (Full Lifecycle Entity)
// ============================================================================

struct Organism {
    Genome genome;
    MultiHeadCAState ca_state;
    StigmergicField* stigmergy;             // Shared across population

    // Runtime tracking
    float fitness;
    float coherence;
    float effective_rank;
    float behavioral_coords[BEHAVIORAL_DIMS];

    // Execution trace (for archive encoding)
    float segment_expression_trace[NUM_SEGMENTS];
    float segment_activation_freq[NUM_SEGMENTS];

    // RNG state
    curandState rand_state;

    // Parent information
    uint32_t parent_a_id;
    uint32_t parent_b_id;
    uint32_t organism_id;
    uint32_t generation;
};

// ============================================================================
// Evolution Context (Shared State)
// ============================================================================

struct EvolutionContext {
    Organism* population;                    // [POPULATION_SIZE]
    Archive archive;
    PopulationMetrics metrics;
    StigmergicField stigmergy;

    // Global counters
    int* next_organism_id;
    int* next_elite_id;
    int current_generation;
};

#endif // TYPES_CUH
