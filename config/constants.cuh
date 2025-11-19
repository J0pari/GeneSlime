#ifndef CONSTANTS_CUH
#define CONSTANTS_CUH

// Grid and spatial resolution
constexpr int GRID_SIZE = 64;
constexpr int GRID_SIZE_Z = 16;
constexpr int GRID_CELLS_2D = GRID_SIZE * GRID_SIZE;
constexpr int GRID_CELLS_3D = GRID_SIZE * GRID_SIZE * GRID_SIZE_Z;

// Multi-head cellular automata
constexpr int NUM_HEADS = 8;
constexpr int CHANNELS = 16;
constexpr int HIDDEN_DIM = 64;
constexpr int HEAD_DIM = 16;

// WMMA tensor core configuration (for Ampere/Ada architectures)
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Genome structure
constexpr int NUM_SEGMENTS = 16;
constexpr int SEGMENT_PAYLOAD_SIZE = 64;
constexpr int TOTAL_GENOME_WEIGHTS = NUM_SEGMENTS * SEGMENT_PAYLOAD_SIZE; // 1024

// Segment permission levels
enum PermissionLevel : uint8_t {
    LEVEL_0_PROTECTED = 0,    // Core infrastructure (0.1% mutation)
    LEVEL_1_STRUCTURAL = 1,   // Head assignment, routing (crisis-only)
    LEVEL_2_PARAMETRIC = 2,   // Weights, biases (100% mutation)
    LEVEL_3_REGULATORY = 3    // Assembly probabilities (1000% amplified)
};

// Stochastic assembly variants
constexpr int NUM_VARIANTS = 3;
constexpr float VARIANT_A_BASE_PROB = 0.70f; // Conservative
constexpr float VARIANT_B_BASE_PROB = 0.25f; // Exploratory
constexpr float VARIANT_C_BASE_PROB = 0.05f; // Disruptive

// Stigmergic substrate timescales
constexpr int NUM_STIGMERGY_LAYERS = 4;
constexpr float DECAY_FAST = 0.9f;      // τ=1 generation
constexpr float DECAY_MEDIUM = 0.99f;   // τ=10 generations
constexpr float DECAY_SLOW = 0.999f;    // τ=100 generations
constexpr float DECAY_STRUCTURAL = 1.0f; // τ=∞ (no decay)

// Threshold-gated write parameters
constexpr float THRESHOLD_FAST = 0.01f;
constexpr float THRESHOLD_MEDIUM = 0.1f;
constexpr float THRESHOLD_SLOW = 0.5f;
constexpr float THRESHOLD_STRUCTURAL = 2.0f;

// Write gain per layer
constexpr float WRITE_GAIN_FAST = 0.3f;
constexpr float WRITE_GAIN_MEDIUM = 0.2f;
constexpr float WRITE_GAIN_SLOW = 0.1f;
constexpr float WRITE_GAIN_STRUCTURAL = 0.05f;

// Flow-Lenia parameters
constexpr float ALPHA_AFFINITY = 0.5f;  // Affinity vs mass gradient balance
constexpr float BETA_GROWTH = 0.3f;     // Growth kernel parameter
constexpr float FLOW_DT = 0.1f;         // Integration timestep
constexpr float FLOW_SIGMA = 1.0f;      // Gaussian overlap sigma
constexpr int N_FLOW_POWER = 2;         // Growth function exponent

// Archive and MAP-Elites
constexpr int MAX_ARCHIVE_SIZE = 10000;
constexpr int MAX_ARCHIVE_CELLS = 1024;
constexpr int NUM_VORONOI_CELLS = 1024;
constexpr int BEHAVIOR_DIM = 10;
constexpr int BEHAVIORAL_DIMS = 10;

// Population and evolution
constexpr int MAX_POPULATION_SIZE = 1024;
constexpr int POPULATION_SIZE = 256;
constexpr int GENERATIONS = 100000;
constexpr int FITNESS_EVAL_INTERVAL = 100;

// Mutation rates by permission level
constexpr float MUTATION_RATE_L0 = 0.001f;  // 0.1%
constexpr float MUTATION_RATE_L1 = 0.1f;    // 10% (crisis only)
constexpr float MUTATION_RATE_L2 = 1.0f;    // 100%
constexpr float MUTATION_RATE_L3 = 10.0f;   // 1000%

// Lifecycle crisis thresholds
constexpr float CRISIS_COHERENCE_THRESHOLD = 0.2f;
constexpr float CRISIS_SATURATION_THRESHOLD = 0.3f;
constexpr float CRISIS_VARIANCE_THRESHOLD = 0.1f;

// Block and thread configurations
constexpr int BLOCK_SIZE_1D = 256;
constexpr int BLOCK_SIZE_2D = 16;
constexpr int WARP_SIZE = 32;

// SVD and fitness computation
constexpr int SVD_MAX_ITERATIONS = 50;
constexpr float SVD_TOLERANCE = 1e-6f;
constexpr int CORRELATION_WINDOW = 100; // Timesteps for correlation matrix

// Compression and trace encoding
constexpr int MAX_PATTERNS = 256;
constexpr int MAX_WRITE_EVENTS = 10000;
constexpr int MAX_TIMESTEPS = 1000;

// Curand RNG
constexpr unsigned long long CURAND_SEED = 42ULL;

#endif // CONSTANTS_CUH
