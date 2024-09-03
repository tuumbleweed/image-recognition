#include <cuda_runtime.h>
#include <stdio.h>

#include <iostream>

#include "io/colors.h"
#include "gpu.h"
#include "io/log.h"

#define GREY "\033[90m"
#define RESET "\033[0m"

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

// Sigmoid derivative for GPU
__device__ float sigmoid_derivative_gpu(float x) {
    // Apply sigmoid to get the output
    float sigmoid_output = sigmoid_gpu(x);
    // Derivative of sigmoid is sigmoid(x) * (1 - sigmoid(x))
    return sigmoid_output * (1.0f - sigmoid_output);
}

__global__ void compute_delta_output_gpu(const float* output, const float* actual, float* delta, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        delta[idx] = (output[idx] - actual[idx]) * sigmoid_derivative_gpu(output[idx]);
        // printf("Delta %d: (%.6f-%.6f) * %.6f = %.6f\n", idx, output[idx], actual[idx], sigmoid_derivative_gpu(output[idx]), delta[idx]);
    }
}

__global__ void update_weights_biases_gpu(float* weights, float* biases, const float* prev_output, const float* delta, int current_size, int prev_size, float learning_rate, int layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < current_size) {
        // printf("Changing weights and biases for layer %d %s\n", layer, GREY);
        for (int k = 0; k < prev_size; ++k) {
            weights[idx * prev_size + k] -= learning_rate * delta[idx] * prev_output[k];
        }
        biases[idx] -= learning_rate * delta[idx];
        // printf("%s\n", RESET);
    }
}

__global__ void compute_delta_hidden_gpu(const float* next_delta, const float* next_weights, const float* output, float* delta, int current_size, int next_size) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < current_size) {
        // printf("Delta %d: (", j);
        float sum = 0.0f;
        for (int k = 0; k < next_size; ++k) {
            sum += next_delta[k] * next_weights[k * current_size + j];
            // printf("%.6f*%.6f", next_delta[k], next_weights[k * current_size + j]);
            // if (k < next_size-1) {
            //     printf(" + ");
            // }
        }
        delta[j] = sum * sigmoid_derivative_gpu(output[j]);
        // printf(") * %.6f = %.6f\n", sigmoid_derivative_gpu(output[j]), delta[j]);
    }
}

__global__ void apply_sigmoid_gpu(float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input[idx] = sigmoid_gpu(input[idx]);
    }
}

__global__ void matrix_multiply_gpu(const float* A, const float* B, const float* biases, float* result, int a_length_b_width, int b_length) {
    int b_row = blockIdx.x * blockDim.x + threadIdx.x;

    if (b_row < b_length) {
        float sum = 0.0f;
        for (int ab_index = 0; ab_index < a_length_b_width; ++ab_index) {
            sum += A[ab_index] * B[b_row * a_length_b_width + ab_index];
        }
        result[b_row] = sum + biases[b_row];
    }
}

std::vector<GPULayer> allocate_network_data_on_gpu(std::vector<Layer>& network) {
    // Allocate and copy weights for each layer to the GPU
    LOG(SL::INFO) << "Allocating data on gpu" << std::endl;
    std::vector<GPULayer> gpu_network(network.size());
    for (size_t i = 0; i < network.size(); ++i) {
        int layer_input_size = network[i].weights[0].size();
        int layer_output_size = network[i].weights.size();
        size_t mem_size_weights = layer_input_size * layer_output_size * sizeof(float);
        size_t mem_size_biases = layer_output_size * sizeof(float);

        // Allocate memory for weights
        cudaMallocWrapper(reinterpret_cast<void**>(&gpu_network[i].weights), mem_size_weights);
        cudaMallocWrapper(reinterpret_cast<void**>(&gpu_network[i].biases), mem_size_biases);
        cudaMallocWrapper(reinterpret_cast<void**>(&gpu_network[i].output), mem_size_biases);
        // Copy weights to the GPU
        std::vector<float> flattened_weights;
        for (const std::vector<float>& row : network[i].weights) {
            flattened_weights.insert(flattened_weights.end(), row.begin(), row.end());
        }
        cudaMemcpyWrapper(gpu_network[i].weights, flattened_weights.data(), mem_size_weights, cudaMemcpyHostToDevice);
        cudaMemcpyWrapper(gpu_network[i].biases, network[i].biases.data(), mem_size_biases, cudaMemcpyHostToDevice);
        cudaMemcpyWrapper(gpu_network[i].output, network[i].output.data(), mem_size_biases, cudaMemcpyHostToDevice);
    }

    return gpu_network;
}

void deallocate_network_data_on_gpu(std::vector<GPULayer>& gpu_network) {
    // Allocate and copy weights for each layer to the GPU
    LOG(SL::INFO) << "Deallocating data on gpu" << std::endl;
    for (size_t i = 0; i < gpu_network.size(); ++i) {
        cudaFreeWrapper(gpu_network[i].weights);
        cudaFreeWrapper(gpu_network[i].biases);
        cudaFreeWrapper(gpu_network[i].output);
    }
}

// Function to initialize memory, copy data, and run forward propagation
void forward_propagation_gpu(const std::vector<float>& input, std::vector<Layer>& network, std::vector<GPULayer>& gpu_network) {
    int input_size = input.size();

    float* d_input;
    // Allocate memory on the GPU
    cudaMallocWrapper(reinterpret_cast<void**>(&d_input), input_size * sizeof(float));
    // Copy the input data to the GPU
    cudaMemcpyWrapper(d_input, input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);

    // Forward propagation through the layers
    float* current_input = d_input;
    LOG(SL::INFO) << "(GPU) Forward Propagation:" << std::endl;
    LOG(SL::DEBUG) << "------------------------------------ Activation Output (GPU) ------------------------------------"
              << std::endl;
    for (size_t i = 0; i < network.size(); ++i) {
        int layer_output_size = network[i].weights.size();

        // Launch matrix multiplication kernel
        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((layer_output_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (input_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
        matrix_multiply_gpu<<<blocksPerGrid, threadsPerBlock>>>(current_input, gpu_network[i].weights, gpu_network[i].biases, gpu_network[i].output, input_size,
                                                                layer_output_size);

        // Debug: Check output after matrix multiplication
        std::vector<float> temp_output(layer_output_size);
        cudaMemcpyWrapper(temp_output.data(), gpu_network[i].output, layer_output_size * sizeof(float),
                          cudaMemcpyDeviceToHost);

        LOG(SL::DEBUG) << "Layer " << i << " Pre-Activation Output (GPU):" << GREY << std::endl;
        for (int j = 0; j < layer_output_size; ++j) {
            LOG(SL::DEBUG) << temp_output[j] << " ";
        }
        LOG(SL::DEBUG) << RESET << std::endl;

        // Apply Sigmoid activation function
        apply_sigmoid_gpu<<<(layer_output_size + 255) / 256, 256>>>(gpu_network[i].output, layer_output_size);

        // Debug: Check output after activation
        cudaMemcpyWrapper(temp_output.data(), gpu_network[i].output, layer_output_size * sizeof(float),
                          cudaMemcpyDeviceToHost);

        LOG(SL::DEBUG) << "Layer " << i << " Post-Activation Output (GPU):" << GREY << std::endl;
        for (int j = 0; j < layer_output_size; ++j) {
            LOG(SL::DEBUG) << temp_output[j] << " ";
        }
        LOG(SL::DEBUG) << RESET << std::endl;

        current_input = gpu_network[i].output;  // Update input for next layer
        input_size = layer_output_size;
        network[i].output = temp_output;
    }
    cudaFreeWrapper(d_input);
}

void backpropagation_gpu(const std::vector<float>& input, const std::vector<float>& actual, 
                         std::vector<Layer>& network, std::vector<GPULayer>& gpu_network, float learning_rate) {
    float* gpu_input;
    float* gpu_actual;
    // Allocate memory on the GPU
    cudaMallocWrapper(reinterpret_cast<void**>(&gpu_input), input.size() * sizeof(float));
    cudaMallocWrapper(reinterpret_cast<void**>(&gpu_actual), actual.size() * sizeof(float));
    // Copy input and actual data to GPU
    cudaMemcpyWrapper(gpu_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyWrapper(gpu_actual, actual.data(), actual.size() * sizeof(float), cudaMemcpyHostToDevice);

    float* delta;
    float* next_delta;
    size_t max_output_size = actual.size(); // Start with the size for the output layer
    for (int i = network.size() - 2; i >= 0; --i) {
        if (network[i].output.size() > max_output_size) {
            max_output_size = network[i].output.size();
        }
    }
    cudaMallocWrapper(reinterpret_cast<void**>(&delta), max_output_size * sizeof(float));
    cudaMallocWrapper(reinterpret_cast<void**>(&next_delta), max_output_size * sizeof(float));

    LOG(SL::INFO) << "(GPU) Backpropagation:" << std::endl;

    // Backward propagation through the layers
    for (int i = network.size() - 1; i >= 0; --i) {
        // Allocate GPU memory for the delta (error term)
        const std::vector<float>& output = network[i].output;
        const std::vector<float>& previous_output = (i == 0) ? input : network[i - 1].output;
        float* gpu_previous_output = (i == 0) ? gpu_input : gpu_network[i - 1].output;

        LOG(SL::DEBUG) << "Computing deltas for layer " << i << " (GPU)" << GREY << std::endl;
        if (i == network.size() - 1) {
            // For output layer
            compute_delta_output_gpu<<<(output.size() + 255) / 256, 256>>>(gpu_network[i].output, gpu_actual, delta, output.size());
        } else {
            cudaMemcpyWrapper(next_delta, delta, max_output_size * sizeof(float), cudaMemcpyDeviceToDevice);
            compute_delta_hidden_gpu<<<(output.size() + 255) / 256, 256>>>(next_delta, gpu_network[i + 1].weights, gpu_network[i].output, delta, output.size(), network[i + 1].output.size());
            // for (int j = 0; j < output.size(); ++j) {
            //     // Launch the kernel with only one thread and one block
            //     compute_delta_hidden_gpu<<<1, 1>>>(next_delta, gpu_network[i + 1].weights, gpu_network[i].output, delta, output.size(), network[i + 1].output.size(), j);
            //     // Ensure that the GPU has finished computing before moving to the next iteration
            //     cudaDeviceSynchronize();
            // }
        }
        // Wait for the GPU to finish before moving on
        cudaDeviceSynchronize();
        LOG(SL::DEBUG) << RESET << std::endl;
 
        // LOG(SL::DEBUG) << "Changing weights and biases for layer " << i << ":" << std::endl;
        // Update weights and biases
        update_weights_biases_gpu<<<(output.size() + 255) / 256, 256>>>(gpu_network[i].weights, gpu_network[i].biases, gpu_previous_output, delta, output.size(), previous_output.size(), learning_rate, i);
        // Wait for the GPU to finish before moving on
        cudaDeviceSynchronize();
        // LOG(SL::DEBUG) << RESET << std::endl;
    }

    // Free GPU memory
    cudaFreeWrapper(gpu_input);
    cudaFreeWrapper(gpu_actual);
    cudaFreeWrapper(delta);
    cudaFreeWrapper(next_delta);
}

// just copy it back to cpu to print later
void network_from_gpu_to_host(std::vector<Layer>& network, std::vector<GPULayer>& gpu_network) {
    for (size_t i = 0; i < network.size(); ++i) {
        Layer& cpu_layer = network[i];
        GPULayer& gpu_layer = gpu_network[i];

        size_t input_dim = cpu_layer.weights[0].size();
        size_t output_dim = cpu_layer.weights.size();

        // Copy weights from GPU to CPU
        for (size_t j = 0; j < output_dim; ++j) {
            cudaMemcpyWrapper(cpu_layer.weights[j].data(), 
                              gpu_layer.weights + j * input_dim, 
                              input_dim * sizeof(float), 
                              cudaMemcpyDeviceToHost);
        }

        // Copy biases from GPU to CPU
        cudaMemcpyWrapper(cpu_layer.biases.data(), gpu_layer.biases, 
                          output_dim * sizeof(float), cudaMemcpyDeviceToHost);

        // Copy output from GPU to CPU
        cudaMemcpyWrapper(cpu_layer.output.data(), gpu_layer.output, 
                          output_dim * sizeof(float), cudaMemcpyDeviceToHost);
    }
}
