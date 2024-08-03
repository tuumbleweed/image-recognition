#include "gpu.h"
#include "colors.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
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

__global__ void matrix_multiply_gpu(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float value = 0.0f;
        for (int e = 0; e < N; ++e) {
            value += A[row * N + e] * B[e * K + col];
        }
        C[row * K + col] = value;
    }
}

// Function to initialize memory, copy data, and run forward propagation
void forward_propagation_gpu(const std::vector<float>& input,
                             const std::vector<Layer>& network,
                             std::vector<float>& output) {
    int input_size = input.size();
    int output_size = network.back().weights.size();

    // Allocate memory on the GPU
    float* d_input;
    float* d_output;
    cudaError_t err;

    std::cout << "------------------------------------Memory Allocation (GPU)------------------------------------" << std::endl;
    // Allocate memory for input
    std::cout << "Allocating " << input_size * sizeof(float) << " bytes for input on the GPU." << std::endl;
    err = cudaMalloc(&d_input, input_size * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed for input: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Allocate memory for output
    std::cout << "Allocating " << output_size * sizeof(float) << " bytes for output on the GPU." << std::endl;
    err = cudaMalloc(&d_output, output_size * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed for output: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        return;
    }

    // Allocate and copy weights for each layer to the GPU
    std::vector<float*> d_weights(network.size());
    for (size_t i = 0; i < network.size(); ++i) {
        int layer_input_size = network[i].weights[0].size();
        int layer_output_size = network[i].weights.size();
        size_t mem_size = layer_input_size * layer_output_size * sizeof(float);

        // Debug: Print the sizes to verify correctness
        std::cout << "Layer " << i << " input size: " << layer_input_size 
                  << ", output size: " << layer_output_size << std::endl;
        std::cout << "Allocating " << mem_size << " bytes for weights on the GPU." << std::endl;

        // Allocate memory for weights
        err = cudaMalloc(&d_weights[i], mem_size);
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc failed for layer " << i 
                      << ": " << cudaGetErrorString(err) << std::endl;
            for (size_t j = 0; j <= i; ++j) {
                cudaFree(d_weights[j]);
            }
            cudaFree(d_input);
            cudaFree(d_output);
            return;
        }

        // Copy weights to the GPU
        err = cudaMemcpy(d_weights[i], network[i].weights[0].data(), mem_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpy failed for layer " << i 
                      << ": " << cudaGetErrorString(err) << std::endl;
            for (size_t j = 0; j <= i; ++j) {
                cudaFree(d_weights[j]);
            }
            cudaFree(d_input);
            cudaFree(d_output);
            return;
        } else {
            std::cout << "cudaMemcpy successful for layer " << i << std::endl;
        }
    }

    // Copy the input data to the GPU
    std::cout << "Copying input data to the GPU." << std::endl;
    err = cudaMemcpy(d_input, input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed for input: " << cudaGetErrorString(err) << std::endl;
        for (auto& d_weight : d_weights) {
            cudaFree(d_weight);
        }
        cudaFree(d_input);
        cudaFree(d_output);
        return;
    }
    std::cout << std::endl;

    // Forward propagation through the layers
    float* current_input = d_input;
    float* current_output = nullptr;

    std::cout << "------------------------------------Activation Output (GPU)------------------------------------" << std::endl;
    for (size_t i = 0; i < network.size(); ++i) {
        int layer_output_size = network[i].weights.size();
        cudaMalloc(&current_output, layer_output_size * sizeof(float));

        // Launch matrix multiplication kernel
        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((layer_output_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (input_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
        matrix_multiply_gpu<<<blocksPerGrid, threadsPerBlock>>>(current_input, d_weights[i], current_output, 1, input_size, layer_output_size);

        // Debug: Check output after matrix multiplication
        std::vector<float> temp_output(layer_output_size);
        cudaMemcpy(temp_output.data(), current_output, layer_output_size * sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "Layer " << i << " Pre-Activation Output (GPU):" << GREY << std::endl;
        for (int j = 0; j < layer_output_size; ++j) {
            std::cout << temp_output[j] << " ";
        }
        std::cout << RESET << std::endl;

        // Apply Sigmoid activation function
        apply_sigmoid_gpu<<<(layer_output_size + 255) / 256, 256>>>(current_output, layer_output_size);

        // Debug: Check output after activation
        cudaMemcpy(temp_output.data(), current_output, layer_output_size * sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "Layer " << i << " Post-Activation Output (GPU):" << GREY << std::endl;
        for (int j = 0; j < layer_output_size; ++j) {
            std::cout << temp_output[j] << " ";
        }
        std::cout << RESET << std::endl;

        current_input = current_output; // Update input for next layer
    }

    // Copy the result back to the host
    std::cout << "Copying output data back to the host." << std::endl;
    output.resize(output_size);
    err = cudaMemcpy(output.data(), current_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed for output: " << cudaGetErrorString(err) << std::endl;
    }

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(current_output);
    for (auto& d_weight : d_weights) {
        cudaFree(d_weight);
    }
}
