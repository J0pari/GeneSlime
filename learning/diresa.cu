
#ifndef DIRESA_CU
#define DIRESA_CU

#include "../config/constants.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>

constexpr int HARDWARE_FEATURES_DIM = TOTAL_GENOME_WEIGHTS;
constexpr int MAX_EMBED_DIM = BEHAVIOR_DIM;

struct DIRESAWeights {
    float* encoder;
    float* decoder;
    int current_dim;
    float reconstruction_error;
};

__global__ void init_diresa_kernel(DIRESAWeights* weights, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {
        weights->current_dim = BEHAVIORAL_DIM;
        weights->reconstruction_error = 0.0f;
    }

    if (tid < HARDWARE_FEATURES_DIM * MAX_EMBED_DIM) {
        curandState state;
        curand_init(seed + tid, 0, 0, &state);
        float scale = sqrtf(2.0f / (HARDWARE_FEATURES_DIM + MAX_EMBED_DIM));
        weights->encoder[tid] = curand_normal(&state) * scale;
    }

    if (tid < MAX_EMBED_DIM * HARDWARE_FEATURES_DIM) {
        curandState state;
        curand_init(seed + tid + BEHAVIOR_DIM * 100000, 0, 0, &state);
        float scale = sqrtf(2.0f / (MAX_EMBED_DIM + HARDWARE_FEATURES_DIM));
        weights->decoder[tid] = curand_normal(&state) * scale;
    }
}

__device__ void diresa_encode(float* hardware_features, float* behavioral_coords, DIRESAWeights* weights) {
    int dim = weights->current_dim;
    for (int i = 0; i < dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < HARDWARE_FEATURES_DIM; j++) {
            sum += hardware_features[j] * weights->encoder[j * MAX_EMBED_DIM + i];
        }
        behavioral_coords[i] = tanhf(sum);
    }
}

__device__ void diresa_decode(float* behavioral_coords, float* reconstructed_features, DIRESAWeights* weights) {
    int dim = weights->current_dim;
    for (int i = 0; i < HARDWARE_FEATURES_DIM; i++) {
        float sum = 0.0f;
        for (int j = 0; j < dim; j++) {
            sum += behavioral_coords[j] * weights->decoder[j * HARDWARE_FEATURES_DIM + i];
        }
        reconstructed_features[i] = sum;
    }
}

__global__ void diresa_reconstruction_loss_kernel(
    float* hardware_features,
    float* behavioral_coords,
    DIRESAWeights* weights,
    float* loss_out,
    int num_samples
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_samples) return;

    float* features = hardware_features + tid * HARDWARE_FEATURES_DIM;
    float* coords = behavioral_coords + tid * MAX_EMBED_DIM;

    float reconstructed[HARDWARE_FEATURES_DIM];
    diresa_decode(coords, reconstructed, weights);

    float mse = 0.0f;
    for (int i = 0; i < HARDWARE_FEATURES_DIM; i++) {
        float diff = features[i] - reconstructed[i];
        mse += diff * diff;
    }

    atomicAdd(loss_out, mse / HARDWARE_FEATURES_DIM);
}

__global__ void diresa_triplet_loss_kernel(
    float* behavioral_coords,
    int* triplet_indices,
    float* loss_out,
    int num_triplets,
    float margin
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_triplets) return;

    int anchor_idx = triplet_indices[tid * 3 + 0];
    int positive_idx = triplet_indices[tid * 3 + 1];
    int negative_idx = triplet_indices[tid * 3 + 2];

    float* anchor = behavioral_coords + anchor_idx * MAX_EMBED_DIM;
    float* positive = behavioral_coords + positive_idx * MAX_EMBED_DIM;
    float* negative = behavioral_coords + negative_idx * MAX_EMBED_DIM;

    float dist_pos = 0.0f;
    float dist_neg = 0.0f;

    for (int i = 0; i < MAX_EMBED_DIM; i++) {
        float diff_pos = anchor[i] - positive[i];
        float diff_neg = anchor[i] - negative[i];
        dist_pos += diff_pos * diff_pos;
        dist_neg += diff_neg * diff_neg;
    }

    float loss = fmaxf(0.0f, sqrtf(dist_pos) - sqrtf(dist_neg) + margin);
    atomicAdd(loss_out, loss);
}

#endif
