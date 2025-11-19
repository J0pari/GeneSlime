
#ifndef AUTODIFF_CU
#define AUTODIFF_CU
#include "../config/constants.cuh"
#include "../utils/tile_ops.cuh"
#include <cuda_runtime.h>

constexpr float EPSILON_SMALL = 1e-8f;

enum TapeOp {
    OP_ADD,
    OP_MUL,
    OP_TANH,
    OP_EXP,
    OP_LOG,
    OP_SQRT,
    OP_SIN,
    OP_COS,
    OP_MATMUL,
    OP_REDUCE_SUM,
    OP_REDUCE_MAX
};

struct TapeEntry {
    TapeOp op;
    int output_idx;
    int input1_idx;
    int input2_idx;
    float aux_data;
    int shape_info;
};

struct ADTape {
    TapeEntry* entries;
    int capacity;
    int current_size;
    float* value_buffer;
    float* grad_buffer;
    int value_capacity;
    int current_value_idx;
};

__global__ void init_ad_tape_kernel(ADTape* tape, int tape_capacity, int value_capacity) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        tape->capacity = tape_capacity;
        tape->current_size = 0;
        tape->value_capacity = value_capacity;
        tape->current_value_idx = 0;
    }
}

__device__ int tape_record_unary(ADTape* tape, TapeOp op, int input_idx, float output_value, float aux_data = 0.0f) {
    int entry_idx = atomicAdd(&tape->current_size, 1);
    if (entry_idx >= tape->capacity) {
        return -1;
    }

    int output_idx = atomicAdd(&tape->current_value_idx, 1);
    if (output_idx >= tape->value_capacity) {
        return -1;
    }

    tape->entries[entry_idx].op = op;
    tape->entries[entry_idx].output_idx = output_idx;
    tape->entries[entry_idx].input1_idx = input_idx;
    tape->entries[entry_idx].input2_idx = -1;
    tape->entries[entry_idx].aux_data = aux_data;

    tape->value_buffer[output_idx] = output_value;
    tape->grad_buffer[output_idx] = 0.0f;

    return output_idx;
}

__device__ int tape_record_binary(ADTape* tape, TapeOp op, int input1_idx, int input2_idx, float output_value) {
    int entry_idx = atomicAdd(&tape->current_size, 1);
    if (entry_idx >= tape->capacity) {
        return -1;
    }

    int output_idx = atomicAdd(&tape->current_value_idx, 1);
    if (output_idx >= tape->value_capacity) {
        return -1;
    }

    tape->entries[entry_idx].op = op;
    tape->entries[entry_idx].output_idx = output_idx;
    tape->entries[entry_idx].input1_idx = input1_idx;
    tape->entries[entry_idx].input2_idx = input2_idx;
    tape->entries[entry_idx].aux_data = 0.0f;

    tape->value_buffer[output_idx] = output_value;
    tape->grad_buffer[output_idx] = 0.0f;

    return output_idx;
}

__device__ int ad_add(ADTape* tape, int x_idx, int y_idx) {
    float result = tape->value_buffer[x_idx] + tape->value_buffer[y_idx];
    return tape_record_binary(tape, OP_ADD, x_idx, y_idx, result);
}

__device__ int ad_mul(ADTape* tape, int x_idx, int y_idx) {
    float result = tape->value_buffer[x_idx] * tape->value_buffer[y_idx];
    return tape_record_binary(tape, OP_MUL, x_idx, y_idx, result);
}

__device__ int ad_tanh(ADTape* tape, int x_idx) {
    float x = tape->value_buffer[x_idx];
    float result = tanhf(x);
    return tape_record_unary(tape, OP_TANH, x_idx, result, result);
}

__device__ int ad_exp(ADTape* tape, int x_idx) {
    float x = tape->value_buffer[x_idx];
    float result = expf(x);
    return tape_record_unary(tape, OP_EXP, x_idx, result, result);
}

__device__ int ad_log(ADTape* tape, int x_idx) {
    float x = tape->value_buffer[x_idx];
    float result = logf(x);
    return tape_record_unary(tape, OP_LOG, x_idx, result, x);
}

__device__ int ad_sqrt(ADTape* tape, int x_idx) {
    float x = tape->value_buffer[x_idx];
    float result = sqrtf(x);
    return tape_record_unary(tape, OP_SQRT, x_idx, result, result);
}

__device__ int ad_sin(ADTape* tape, int x_idx) {
    float x = tape->value_buffer[x_idx];
    float result = sinf(x);
    return tape_record_unary(tape, OP_SIN, x_idx, result, x);
}

__device__ int ad_cos(ADTape* tape, int x_idx) {
    float x = tape->value_buffer[x_idx];
    float result = cosf(x);
    return tape_record_unary(tape, OP_COS, x_idx, result, x);
}

__global__ void ad_backward_kernel(ADTape* tape, int output_idx, float output_grad) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    tape->grad_buffer[output_idx] = output_grad;

    for (int i = tape->current_size - 1; i >= 0; i--) {
        TapeEntry* entry = &tape->entries[i];
        float output_grad = tape->grad_buffer[entry->output_idx];

        switch (entry->op) {
            case OP_ADD: {
                tape->grad_buffer[entry->input1_idx] += output_grad;
                tape->grad_buffer[entry->input2_idx] += output_grad;
                break;
            }
            case OP_MUL: {
                float x = tape->value_buffer[entry->input1_idx];
                float y = tape->value_buffer[entry->input2_idx];
                tape->grad_buffer[entry->input1_idx] += output_grad * y;
                tape->grad_buffer[entry->input2_idx] += output_grad * x;
                break;
            }
            case OP_TANH: {
                float tanh_x = entry->aux_data;
                float grad_x = output_grad * (1.0f - tanh_x * tanh_x);
                tape->grad_buffer[entry->input1_idx] += grad_x;
                break;
            }
            case OP_EXP: {
                float exp_x = entry->aux_data;
                tape->grad_buffer[entry->input1_idx] += output_grad * exp_x;
                break;
            }
            case OP_LOG: {
                float x = entry->aux_data;
                tape->grad_buffer[entry->input1_idx] += output_grad / (x + EPSILON_SMALL);
                break;
            }
            case OP_SQRT: {
                float sqrt_x = entry->aux_data;
                tape->grad_buffer[entry->input1_idx] += output_grad / (2.0f * sqrt_x + EPSILON_SMALL);
                break;
            }
            case OP_SIN: {
                float x = entry->aux_data;
                tape->grad_buffer[entry->input1_idx] += output_grad * cosf(x);
                break;
            }
            case OP_COS: {
                float x = entry->aux_data;
                tape->grad_buffer[entry->input1_idx] += -output_grad * sinf(x);
                break;
            }
            default:
                break;
        }
    }
}

__global__ void reset_tape_kernel(ADTape* tape) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < tape->value_capacity) {
        tape->grad_buffer[tid] = 0.0f;
    }

    if (tid == 0) {
        tape->current_size = 0;
        tape->current_value_idx = 0;
    }
}

__global__ void extract_genome_gradients_kernel(
    ADTape* tape,
    int* genome_param_indices,
    int num_params,
    float* output_gradients
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_params) {
        int param_idx = genome_param_indices[tid];
        output_gradients[tid] = tape->grad_buffer[param_idx];
    }
}

__global__ void apply_gradients_kernel(
    float* genome,
    float* gradients,
    int genome_size,
    float learning_rate,
    float gradient_clip_norm
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < genome_size) {
        float grad = gradients[tid];

        if (fabsf(grad) > gradient_clip_norm) {
            grad = copysignf(gradient_clip_norm, grad);
        }

        genome[tid] -= learning_rate * grad;

        genome[tid] = clamp(genome[tid], -1.0f, 1.0f);
    }
}

#endif
