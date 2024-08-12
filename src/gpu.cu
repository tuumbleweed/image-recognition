#include <cuda_runtime.h>
#include <stdio.h>

#include <iostream>

#include "colors.h"
#include "gpu.h"

extern void* cudaMallocWrapper(void** ptr, size_t size);
extern void cudaMemcpyWrapper(void* dst, const void* src, size_t size, cudaMemcpyKind kind);
extern void cudaFreeWrapper(void* ptr);

#define CUDA_CHECK(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__device__ float sigmoid_gpu(float x) {
    if (x >= 0) {
        return 1.0f / (1.0f + expf(-x));
    } else {
        float exp_x = expf(x);
        return exp_x / (1.0f + exp_x);
    }
}

__global__ void apply_sigmoid_gpu(float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input[idx] = sigmoid_gpu(input[idx]);
    }
}

__global__ void matrix_multiply_gpu(const float* A, const float* B, float* result, int b_length, int b_width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < 1 && col < b_width) {
        float sum = 0.0f;
        for (int e = 0; e < b_length; ++e) {
            sum += A[row * b_length + e] * B[e * b_width + col];
        }
        result[row * b_width + col] = sum;
    }
}

// Function to initialize memory, copy data, and run forward propagation
void forward_propagation_gpu(const std::vector<float>& input, const std::vector<Layer>& network,
                             std::vector<float>& output) {
    int input_size = input.size();
    int output_size = network.back().weights.size();

    // Allocate memory on the GPU
    float* d_input;
    float* d_output;

    // Allocate memory for input
    cudaMallocWrapper(reinterpret_cast<void**>(&d_input), input_size * sizeof(float));
    // Allocate memory for output
    cudaMallocWrapper(reinterpret_cast<void**>(&d_output), output_size * sizeof(float));

    // Allocate and copy weights for each layer to the GPU
    std::vector<float*> d_weights(network.size());
    for (size_t i = 0; i < network.size(); ++i) {
        int layer_input_size = network[i].weights[0].size();
        int layer_output_size = network[i].weights.size();
        size_t mem_size = layer_input_size * layer_output_size * sizeof(float);

        // Allocate memory for weights
        cudaMallocWrapper(reinterpret_cast<void**>(&d_weights[i]), mem_size);
        // Copy weights to the GPU
        std::vector<float> flattened_weights;
        for (const std::vector<float>& row : network[i].weights) {
            flattened_weights.insert(flattened_weights.end(), row.begin(), row.end());
        }
        cudaMemcpyWrapper(d_weights[i], flattened_weights.data(), mem_size, cudaMemcpyHostToDevice);
    }

    // Copy the input data to the GPU
    cudaMemcpyWrapper(d_input, input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);
    std::cout << std::endl;

    // Forward propagation through the layers
    float* current_input = d_input;
    float* current_output = nullptr;

    std::cout << "------------------------------------Activation Output (GPU)------------------------------------"
              << std::endl;
    for (size_t i = 0; i < network.size(); ++i) {
        int layer_output_size = network[i].weights.size();

        // Allocate memory for current_output
        cudaMallocWrapper(reinterpret_cast<void**>(&current_output), layer_output_size * sizeof(float));

        // Launch matrix multiplication kernel
        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((layer_output_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (input_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
        matrix_multiply_gpu<<<blocksPerGrid, threadsPerBlock>>>(current_input, d_weights[i], current_output, input_size,
                                                                layer_output_size);

        // Debug: Check output after matrix multiplication
        std::vector<float> temp_output(layer_output_size);
        cudaMemcpyWrapper(temp_output.data(), current_output, layer_output_size * sizeof(float),
                          cudaMemcpyDeviceToHost);

        std::cout << "Layer " << i << " Pre-Activation Output (GPU):" << GREY << std::endl;
        for (int j = 0; j < layer_output_size; ++j) {
            std::cout << temp_output[j] << " ";
        }
        std::cout << RESET << std::endl;

        // Apply Sigmoid activation function
        apply_sigmoid_gpu<<<(layer_output_size + 255) / 256, 256>>>(current_output, layer_output_size);

        // Debug: Check output after activation
        cudaMemcpyWrapper(temp_output.data(), current_output, layer_output_size * sizeof(float),
                          cudaMemcpyDeviceToHost);

        std::cout << "Layer " << i << " Post-Activation Output (GPU):" << GREY << std::endl;
        for (int j = 0; j < layer_output_size; ++j) {
            std::cout << temp_output[j] << " ";
        }
        std::cout << RESET << std::endl;

        current_input = current_output;  // Update input for next layer
    }

    // Copy the result back to the host
    output.resize(output_size);
    cudaMemcpyWrapper(output.data(), current_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFreeWrapper(d_input);
    cudaFreeWrapper(current_output);
    for (auto& d_weight : d_weights) {
        cudaFreeWrapper(d_weight);
    }
}
