#ifndef DIRESA_TRAINING_CU
#define DIRESA_TRAINING_CU

#include "../config/constants.cuh"
#include "diresa.cu"
#include "autodiff.cu"
#include <cuda_runtime.h>

__global__ void extract_genome_features_kernel(
    Organism* population,
    int population_size,
    float* genome_features
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= population_size) return;

    float* features = genome_features + tid * TOTAL_GENOME_WEIGHTS;

    for (int seg = 0; seg < NUM_SEGMENTS; seg++) {
        for (int w = 0; w < SEGMENT_PAYLOAD_SIZE; w++) {
            int idx = seg * SEGMENT_PAYLOAD_SIZE + w;
            features[idx] = population[tid].genome.segments[seg].payload[w];
        }
    }
}

__global__ void diresa_forward_batch_kernel(
    float* genome_features,
    float* behavioral_coords,
    DIRESAWeights* weights,
    int batch_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    float* features = genome_features + tid * TOTAL_GENOME_WEIGHTS;
    float* coords = behavioral_coords + tid * BEHAVIOR_DIM;

    diresa_encode(features, coords, weights);
}

__global__ void diresa_update_weights_kernel(
    DIRESAWeights* weights,
    float* encoder_grads,
    float* decoder_grads,
    float learning_rate,
    float gradient_clip_norm
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int encoder_size = TOTAL_GENOME_WEIGHTS * BEHAVIOR_DIM;
    int decoder_size = BEHAVIOR_DIM * TOTAL_GENOME_WEIGHTS;

    if (tid < encoder_size) {
        float grad = encoder_grads[tid];
        if (fabsf(grad) > gradient_clip_norm) {
            grad = copysignf(gradient_clip_norm, grad);
        }
        weights->encoder[tid] -= learning_rate * grad;
    }

    if (tid < decoder_size) {
        float grad = decoder_grads[tid];
        if (fabsf(grad) > gradient_clip_norm) {
            grad = copysignf(gradient_clip_norm, grad);
        }
        weights->decoder[tid] -= learning_rate * grad;
    }
}

__global__ void compute_combined_loss_kernel(
    float* genome_features,
    float* behavioral_coords,
    DIRESAWeights* weights,
    int* triplet_indices,
    int num_samples,
    int num_triplets,
    float* recon_loss_out,
    float* triplet_loss_out,
    float triplet_margin
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_samples) {
        float* features = genome_features + tid * TOTAL_GENOME_WEIGHTS;
        float* coords = behavioral_coords + tid * BEHAVIOR_DIM;

        float reconstructed[TOTAL_GENOME_WEIGHTS];
        diresa_decode(coords, reconstructed, weights);

        float mse = 0.0f;
        for (int i = 0; i < TOTAL_GENOME_WEIGHTS; i++) {
            float diff = features[i] - reconstructed[i];
            mse += diff * diff;
        }
        atomicAdd(recon_loss_out, mse / TOTAL_GENOME_WEIGHTS);
    }

    if (tid < num_triplets) {
        int anchor_idx = triplet_indices[tid * 3 + 0];
        int positive_idx = triplet_indices[tid * 3 + 1];
        int negative_idx = triplet_indices[tid * 3 + 2];

        float* anchor = behavioral_coords + anchor_idx * BEHAVIOR_DIM;
        float* positive = behavioral_coords + positive_idx * BEHAVIOR_DIM;
        float* negative = behavioral_coords + negative_idx * BEHAVIOR_DIM;

        float dist_pos = 0.0f;
        float dist_neg = 0.0f;

        for (int i = 0; i < BEHAVIOR_DIM; i++) {
            float diff_pos = anchor[i] - positive[i];
            float diff_neg = anchor[i] - negative[i];
            dist_pos += diff_pos * diff_pos;
            dist_neg += diff_neg * diff_neg;
        }

        float loss = fmaxf(0.0f, sqrtf(dist_pos) - sqrtf(dist_neg) + triplet_margin);
        atomicAdd(triplet_loss_out, loss);
    }
}

__global__ void sample_triplets_kernel(
    Organism* population,
    int population_size,
    int* triplet_indices,
    int num_triplets,
    curandState* rand_states
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_triplets) return;

    curandState state = rand_states[tid];

    int anchor_idx = curand(&state) % population_size;
    int positive_idx = curand(&state) % population_size;
    int negative_idx = curand(&state) % population_size;

    while (positive_idx == anchor_idx) {
        positive_idx = curand(&state) % population_size;
    }
    while (negative_idx == anchor_idx || negative_idx == positive_idx) {
        negative_idx = curand(&state) % population_size;
    }

    triplet_indices[tid * 3 + 0] = anchor_idx;
    triplet_indices[tid * 3 + 1] = positive_idx;
    triplet_indices[tid * 3 + 2] = negative_idx;

    rand_states[tid] = state;
}

void train_diresa_step(
    Archive* d_archive,
    Organism* d_population,
    int population_size,
    curandState* d_rand_states,
    float learning_rate,
    float triplet_margin
) {
    DIRESAWeights* d_diresa_weights;
    cudaMemcpy(&d_diresa_weights, &d_archive, sizeof(DIRESAWeights*), cudaMemcpyDeviceToHost);

    float *d_genome_features, *d_behavioral_coords;
    cudaMalloc(&d_genome_features, population_size * TOTAL_GENOME_WEIGHTS * sizeof(float));
    cudaMalloc(&d_behavioral_coords, population_size * BEHAVIOR_DIM * sizeof(float));

    int threads = 256;
    int blocks = (population_size + threads - 1) / threads;

    extract_genome_features_kernel<<<blocks, threads>>>(
        d_population,
        population_size,
        d_genome_features
    );
    cudaDeviceSynchronize();

    diresa_forward_batch_kernel<<<blocks, threads>>>(
        d_genome_features,
        d_behavioral_coords,
        d_diresa_weights,
        population_size
    );
    cudaDeviceSynchronize();

    int num_triplets = population_size / 2;
    int *d_triplet_indices;
    cudaMalloc(&d_triplet_indices, num_triplets * 3 * sizeof(int));

    int triplet_blocks = (num_triplets + threads - 1) / threads;
    sample_triplets_kernel<<<triplet_blocks, threads>>>(
        d_population,
        population_size,
        d_triplet_indices,
        num_triplets,
        d_rand_states
    );
    cudaDeviceSynchronize();

    float *d_recon_loss, *d_triplet_loss;
    cudaMalloc(&d_recon_loss, sizeof(float));
    cudaMalloc(&d_triplet_loss, sizeof(float));
    cudaMemset(d_recon_loss, 0, sizeof(float));
    cudaMemset(d_triplet_loss, 0, sizeof(float));

    // Compute reconstruction loss
    diresa_reconstruction_loss_kernel<<<blocks, threads>>>(
        d_genome_features,
        d_behavioral_coords,
        d_diresa_weights,
        d_recon_loss,
        population_size
    );
    cudaDeviceSynchronize();

    // Compute triplet loss
    diresa_triplet_loss_kernel<<<triplet_blocks, threads>>>(
        d_behavioral_coords,
        d_triplet_indices,
        d_triplet_loss,
        num_triplets,
        triplet_margin
    );
    cudaDeviceSynchronize();

    // Initialize autodiff tape for gradient computation
    ADTape* d_tape;
    cudaMalloc(&d_tape, sizeof(ADTape));
    init_ad_tape_kernel<<<1, 1>>>(d_tape, 100000);
    cudaDeviceSynchronize();

    // Compute gradients via autodiff backward pass
    float *d_encoder_grads, *d_decoder_grads;
    cudaMalloc(&d_encoder_grads, TOTAL_GENOME_WEIGHTS * BEHAVIOR_DIM * sizeof(float));
    cudaMalloc(&d_decoder_grads, BEHAVIOR_DIM * TOTAL_GENOME_WEIGHTS * sizeof(float));

    // Backward pass from reconstruction loss
    ad_backward_kernel<<<1, 1>>>(d_tape, 0, 1.0f);
    cudaDeviceSynchronize();

    // Extract gradients from tape
    extract_genome_gradients_kernel<<<threads, threads>>>(
        d_tape,
        d_encoder_grads,
        d_decoder_grads,
        TOTAL_GENOME_WEIGHTS * BEHAVIOR_DIM,
        BEHAVIOR_DIM * TOTAL_GENOME_WEIGHTS
    );
    cudaDeviceSynchronize();

    int grad_size = fmaxf(TOTAL_GENOME_WEIGHTS * BEHAVIOR_DIM, BEHAVIOR_DIM * TOTAL_GENOME_WEIGHTS);
    int grad_blocks = (grad_size + threads - 1) / threads;

    // Update weights with computed gradients
    apply_gradients_kernel<<<grad_blocks, threads>>>(
        d_diresa_weights->encoder,
        d_encoder_grads,
        learning_rate,
        TOTAL_GENOME_WEIGHTS * BEHAVIOR_DIM
    );
    apply_gradients_kernel<<<grad_blocks, threads>>>(
        d_diresa_weights->decoder,
        d_decoder_grads,
        learning_rate,
        BEHAVIOR_DIM * TOTAL_GENOME_WEIGHTS
    );
    cudaDeviceSynchronize();

    // Cleanup tape
    reset_tape_kernel<<<1, 1>>>(d_tape);
    cudaFree(d_tape);

    cudaFree(d_genome_features);
    cudaFree(d_behavioral_coords);
    cudaFree(d_triplet_indices);
    cudaFree(d_recon_loss);
    cudaFree(d_triplet_loss);
    cudaFree(d_encoder_grads);
    cudaFree(d_decoder_grads);
}

#endif
