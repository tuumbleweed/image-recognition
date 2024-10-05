#include <cuda_runtime.h>
#include <stdio.h>

#include <iostream>

#include "io/colors.h"
#include "io/log.h"
#include "network/gpu.h"

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

__device__ double relu_derivative_gpu(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

__device__ double sigmoid_gpu(double x) {
    if (x >= 0) {
        return 1.0f / (1.0f + expf(-x));
    } else {
        double exp_x = expf(x);
        return exp_x / (1.0f + exp_x);
    }
}

// Sigmoid derivative for GPU
__device__ double sigmoid_derivative_gpu(double x) {
    // Apply sigmoid to get the output
    double sigmoid_output = sigmoid_gpu(x);
    // Derivative of sigmoid is sigmoid(x) * (1 - sigmoid(x))
    return sigmoid_output * (1.0f - sigmoid_output);
}

__global__ void compute_delta_output_gpu(const double* output, const double* actual, double* delta, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        delta[idx] = (output[idx] - actual[idx]) * sigmoid_derivative_gpu(output[idx]);
        // if (idx == 0) {
        //     printf("%f = (%f - %f) * %f\n", delta[idx], output[idx], actual[idx], sigmoid_derivative_gpu(output[idx]));
        // }
    }
}

__global__ void compute_delta_hidden_gpu(const double* next_delta, const double* next_weights, const double* output,
                                         double* delta, int current_size, int next_size) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < current_size) {
        double sum = 0.0f;
        for (int k = 0; k < next_size; ++k) {
            sum += next_delta[k] * next_weights[k * current_size + j];
        }
        // delta[j] = sum * sigmoid_derivative_gpu(output[j]);
        delta[j] = sum * relu_derivative_gpu(output[j]);
    }
}

__global__ void apply_sigmoid_gpu(double* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input[idx] = sigmoid_gpu(input[idx]);
    }
}

__global__ void matrix_multiply_gpu(const double* A, const double* B, const double* biases, double* result,
                                    int a_length_b_width, int b_length) {
    int b_row = blockIdx.x * blockDim.x + threadIdx.x;

    if (b_row < b_length) {
        double sum = 0.0f;
        for (int ab_index = 0; ab_index < a_length_b_width; ++ab_index) {
            sum += A[ab_index] * B[b_row * a_length_b_width + ab_index];
        }
        result[b_row] = sum + biases[b_row];
    }
}


// Kernel for accumulating weight and bias updates
__global__ void accumulate_weight_bias_updates_atomic(double* accumulated_weights, double* accumulated_biases, 
                                               const double* weights, const double* biases, 
                                               const double* deltasI, const double* prev_output,
                                               int current_size, int prev_size, double learning_rate) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < current_size) {
        for (int k = 0; k < prev_size; ++k) {
            // Accumulate weight updates (use atomicAdd if required for multi-thread safety)
            atomicAdd(&accumulated_weights[j * prev_size + k], -learning_rate * deltasI[j] * prev_output[k]);
        }
        // Accumulate bias updates
        atomicAdd(&accumulated_biases[j], -learning_rate * deltasI[j]);
    }
}

// After batch accumulation, apply the accumulated weights and biases
__global__ void apply_accumulated_updates(double* weights, double* biases, double* accumulated_weights, 
                                          double* accumulated_biases, int current_size, int prev_size, int batch_size) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < current_size) {
        for (int k = 0; k < prev_size; ++k) {
            // Apply accumulated weight updates, averaged over the batch size
            weights[j * prev_size + k] += accumulated_weights[j * prev_size + k] / batch_size;
        }
        // Apply accumulated bias updates, averaged over the batch size
        biases[j] += accumulated_biases[j] / batch_size;
    }
}

std::vector<GPULayer> allocate_network_data_on_gpu(std::vector<Layer>& network) {
    // Allocate and copy weights for each layer to the GPU
    LOG(SL::INFO) << "Allocating data on gpu" << std::endl;
    std::vector<GPULayer> gpu_network(network.size());
    for (size_t i = 0; i < network.size(); ++i) {
        int layer_input_size = network[i].weights[0].size();
        int layer_output_size = network[i].weights.size();
        size_t mem_size_weights = layer_input_size * layer_output_size * sizeof(double);
        size_t mem_size_biases = layer_output_size * sizeof(double);

        // Allocate memory for weights
        cudaMallocWrapper(reinterpret_cast<void**>(&gpu_network[i].weights), mem_size_weights);
        cudaMallocWrapper(reinterpret_cast<void**>(&gpu_network[i].biases), mem_size_biases);
        cudaMallocWrapper(reinterpret_cast<void**>(&gpu_network[i].output), mem_size_biases);
        // Copy weights to the GPU
        std::vector<double> flattened_weights;
        for (const std::vector<double>& row : network[i].weights) {
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
vec_double_2d forward_propagation_gpu(const std::vector<double>& input, const std::vector<Layer>& network,
                             std::vector<GPULayer>& gpu_network) {
    int input_size = input.size();
    vec_double_2d activations(network.size());

    double* d_input;
    // Allocate memory on the GPU
    cudaMallocWrapper(reinterpret_cast<void**>(&d_input), input_size * sizeof(double));
    // Copy the input data to the GPU
    cudaMemcpyWrapper(d_input, input.data(), input_size * sizeof(double), cudaMemcpyHostToDevice);

    // Forward propagation through the layers
    double* current_input = d_input;
    LOG(SL::INFO) << "(GPU) Forward Propagation:" << std::endl;
    LOG(SL::DEBUG)
        << "------------------------------------ Activation Output (GPU) ------------------------------------"
        << std::endl;
    for (size_t i = 0; i < network.size(); ++i) {
        int layer_output_size = network[i].weights.size();
        activations[i].resize(layer_output_size);

        // Launch matrix multiplication kernel
        matrix_multiply_gpu<<<(layer_output_size + 15) / 16, 16>>>(current_input, gpu_network[i].weights,
                                                                gpu_network[i].biases, gpu_network[i].output,
                                                                input_size, layer_output_size);

        if (SL::should_log(SL::DEBUG)) {
            // Debug: Check output after matrix multiplication
            cudaMemcpyWrapper(activations[i].data(), gpu_network[i].output, layer_output_size * sizeof(double),
                            cudaMemcpyDeviceToHost);
            LOG(SL::DEBUG) << "Layer " << i << " Pre-Activation Output (GPU):" << GREY << std::endl;
            for (int j = 0; j < layer_output_size; ++j) {
                LOG(SL::DEBUG) << activations[i][j] << " ";
            }
            LOG(SL::DEBUG) << RESET << std::endl;
        }

        // Apply Sigmoid activation function
        apply_sigmoid_gpu<<<(layer_output_size + 15) / 16, 16>>>(gpu_network[i].output, layer_output_size);

        // Debug: Check output after activation
        cudaMemcpyWrapper(activations[i].data(), gpu_network[i].output, layer_output_size * sizeof(double),
                        cudaMemcpyDeviceToHost);
        LOG(SL::DEBUG) << "Layer " << i << " Post-Activation Output (GPU):" << GREY << std::endl;
        for (int j = 0; j < layer_output_size; ++j) {
            LOG(SL::DEBUG) << activations[i][j] << " ";
        }
        LOG(SL::DEBUG) << RESET << std::endl;

        current_input = gpu_network[i].output;  // Update input for next layer
        input_size = layer_output_size;
    }
    cudaFreeWrapper(d_input);

    return activations;
}

void backpropagation_gpu(const vec_double_2d& image_batch, const std::vector<double>& actual, std::vector<Layer>& network,
                         std::vector<GPULayer>& gpu_network, const double learning_rate, const std::vector<vec_double_2d>& all_activations) {

    double* gpu_actual;
    cudaMallocWrapper(reinterpret_cast<void**>(&gpu_actual), actual.size() * sizeof(double));
    cudaMemcpyWrapper(gpu_actual, actual.data(), actual.size() * sizeof(double), cudaMemcpyHostToDevice);

    std::vector<double*> delta(network.size());
    for (size_t i = 0; i < network.size(); i++) {
        cudaMallocWrapper(reinterpret_cast<void**>(&delta[i]), network[i].output.size() * sizeof(double));
    }

    LOG(SL::INFO) << "(GPU) Backpropagation:" << std::endl;
    // Backward propagation through the layers
    for (int i = network.size() - 1; i >= 0; --i) {
        // copy activations average to gpu
        cudaMemcpyWrapper(gpu_network[i].output, network[i].output.data(), network[i].output.size() * sizeof(double),
                        cudaMemcpyHostToDevice);
        const std::vector<double>& output = network[i].output;

        LOG(SL::DEBUG) << "Computing deltas for layer " << i << " (GPU)" << GREY << std::endl;
        if (i == network.size() - 1) {
            // For output layer
            compute_delta_output_gpu<<<(output.size() + 15) / 16, 16>>>(gpu_network[i].output, gpu_actual, delta[i],
                                                                           output.size());
        } else {
            compute_delta_hidden_gpu<<<(output.size() + 15) / 16, 16>>>(delta[i + 1], gpu_network[i + 1].weights,
                                                                           gpu_network[i].output, delta[i], output.size(),
                                                                           network[i + 1].output.size());
        }
        LOG(SL::DEBUG) << RESET << std::endl;
    }

    size_t max_gpu_previous_output_size = image_batch[0].size();
    for (std::vector<double> activation : all_activations[0]) {
        if (max_gpu_previous_output_size < activation.size()) {
            max_gpu_previous_output_size = activation.size();
        }
    }
    max_gpu_previous_output_size *= sizeof(double);
    double* gpu_previous_output;
    cudaMallocWrapper(reinterpret_cast<void**>(&gpu_previous_output), max_gpu_previous_output_size);

    // Allocate memory for accumulated weight and bias changes
    std::vector<double*> accumulated_weights(network.size());
    std::vector<double*> accumulated_biases(network.size());

    for (size_t i = 0; i < network.size(); i++) {
        size_t weight_size = network[i].weights.size() * network[i].weights[0].size() * sizeof(double);
        size_t bias_size = network[i].biases.size() * sizeof(double);

        cudaMallocWrapper(reinterpret_cast<void**>(&accumulated_weights[i]), weight_size);
        cudaMallocWrapper(reinterpret_cast<void**>(&accumulated_biases[i]), bias_size);

        // Initialize to zero
        cudaMemset(accumulated_weights[i], 0.0f, weight_size);
        cudaMemset(accumulated_biases[i], 0.0f, bias_size);
    }

    size_t image_batch_size = image_batch.size();
    for (size_t l = 0; l < image_batch_size; l++) {
    for (int i = network.size() - 1; i >= 0; --i) {
            const std::vector<double>& previous_output = (i == 0) ? image_batch[l] : all_activations[l][i - 1];
            cudaMemcpyWrapper(gpu_previous_output, previous_output.data(), previous_output.size() * sizeof(double), cudaMemcpyHostToDevice);

            // Accumulate weight and bias updates
            accumulate_weight_bias_updates_atomic<<<(network[i].output.size() + 15) / 16, 16>>>(
                accumulated_weights[i], accumulated_biases[i], gpu_network[i].weights, gpu_network[i].biases, 
                delta[i], gpu_previous_output, network[i].output.size(), previous_output.size(), learning_rate);
        }
    }

    // Apply the accumulated updates after all batches are processed
    for (int i = network.size() - 1; i >= 0; --i) {
        const std::vector<double>& previous_output = (i == 0) ? image_batch[0] : all_activations[0][i - 1];
        apply_accumulated_updates<<<(network[i].output.size() + 15) / 16, 16>>>(
            gpu_network[i].weights, gpu_network[i].biases, accumulated_weights[i], accumulated_biases[i], 
            network[i].output.size(), previous_output.size(), image_batch_size);
    }

    // Host vectors to hold the copied data
    std::vector<std::vector<double>> host_delta;
    std::vector<std::vector<double>> host_weights;
    std::vector<std::vector<double>> host_biases;
    // Copy data from GPU to host for inspection
    copy_accumulated_weights_to_host(delta, accumulated_weights, accumulated_biases, host_delta, host_weights, host_biases, network);

    // for (int i = network.size() - 1; i >= 2; --i) {
    //     const std::vector<double>& previous_output = (i == 0) ? image_batch[0] : all_activations[0][i - 1];
    //     for (size_t j = 0; j < network[i].weights.size(); ++j) {
    //             LOG(SL::NOTICE) << "Weights " << j << "x" << network[i].weights.size() << ":" << GREY << std::endl;
    //         for (size_t k = 0; k < previous_output.size(); ++k) {
    //             LOG(SL::NOTICE) << network[i].weights[j][k] << "-=" << host_weights[i][j*previous_output.size()+k] / image_batch_size << " ";
    //         }
    //         LOG(SL::NOTICE) << RESET << std::endl;
    //         LOG(SL::NOTICE) << "Biases:" << GREY << std::endl;
    //         LOG(SL::NOTICE) << network[i].biases[j] << "-=" << host_biases[i][j] / image_batch_size << " ";
    //         LOG(SL::NOTICE) << RESET << std::endl;
    //     }
    //     LOG(SL::NOTICE) << RESET << std::endl;
    // }
    // for (size_t l = 0; l < image_batch_size; l++) {
    //     for (int i = network.size() - 1; i >= 2; --i) {
    //     // for (int i = network.size() - 2; i >= 1; --i) {
    //         const std::vector<double>& previous_output = (i == 0) ? image_batch[l] : all_activations[l][i - 1];
    //         // for (size_t j = 0; j < network[i].weights.size(); ++j) {
    //         for (size_t j = 0; j < 1; ++j) {
    //             LOG(SL::NOTICE) << "Weights " << j << "x" << network[i].weights.size() << ":" << GREY << std::endl;
    //             for (size_t k = 0; k < previous_output.size(); ++k) {
    //                 LOG(SL::NOTICE) << network[i].weights[j][k] << "-=" << learning_rate << "*" << host_delta[i][j] << "*"
    //                                 << previous_output[k] << " ";
    //                 // LOG(SL::NOTICE) << network[i].weights[j][k] << "-=" << host_weights[i][j*previous_output.size()+k] / image_batch_size << " ";
    //             }
    //             LOG(SL::NOTICE) << RESET << std::endl;
    //             LOG(SL::NOTICE) << "Biases:" << GREY << std::endl;
    //             LOG(SL::NOTICE) << network[i].biases[j] << "+=" << learning_rate << "*" << host_delta[i][j] << " ";
    //             // LOG(SL::NOTICE) << network[i].biases[j] << "-=" << host_biases[i][j] / image_batch_size << " ";
    //             LOG(SL::NOTICE) << RESET << std::endl;
    //         }
    //         LOG(SL::NOTICE) << RESET << std::endl;
    //     }
    // }

    // Free GPU memory
    for (size_t i = 0; i < network.size(); i++) {
        cudaFreeWrapper(accumulated_weights[i]);
        cudaFreeWrapper(accumulated_biases[i]);
    }

    cudaFreeWrapper(gpu_previous_output);
    cudaFreeWrapper(gpu_actual);

    for (size_t i = 0; i < network.size(); i++) {
        cudaFreeWrapper(delta[i]);
    }
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
            cudaMemcpyWrapper(cpu_layer.weights[j].data(), gpu_layer.weights + j * input_dim, input_dim * sizeof(double),
                              cudaMemcpyDeviceToHost);
        }

        // Copy biases from GPU to CPU
        cudaMemcpyWrapper(cpu_layer.biases.data(), gpu_layer.biases, output_dim * sizeof(double),
                          cudaMemcpyDeviceToHost);

        // Copy output from GPU to CPU
        cudaMemcpyWrapper(cpu_layer.output.data(), gpu_layer.output, output_dim * sizeof(double),
                          cudaMemcpyDeviceToHost);
    }
}

// Assume network.size() gives the size of the network layers
void copy_accumulated_weights_to_host(
    const std::vector<double*>& delta,
    const std::vector<double*>& accumulated_weights_gpu,
    const std::vector<double*>& accumulated_biases_gpu,
    std::vector<std::vector<double>>& host_delta,
    std::vector<std::vector<double>>& host_weights,
    std::vector<std::vector<double>>& host_biases,
    const std::vector<Layer>& network
) {
    size_t network_size = network.size();
    
    // Resize host vectors to match network size
    host_delta.resize(network_size);
    host_weights.resize(network_size);
    host_biases.resize(network_size);
    
    for (size_t i = 0; i < network_size; ++i) {
        size_t delta_size = network[i].output.size();
        size_t weights_size = network[i].weights.size() * network[i].weights[0].size();
        size_t biases_size = network[i].biases.size();
        
        // Allocate space on host for each layer's weights and biases
        host_delta[i].resize(delta_size);
        host_weights[i].resize(weights_size);
        host_biases[i].resize(biases_size);

        // Copy delta from GPU to host
        cudaMemcpyWrapper(host_delta[i].data(), delta[i], 
                              delta_size * sizeof(double), cudaMemcpyDeviceToHost);
        
        // Copy weights from GPU to host
        cudaMemcpyWrapper(host_weights[i].data(), accumulated_weights_gpu[i], 
                              weights_size * sizeof(double), cudaMemcpyDeviceToHost);
        
        // Copy biases from GPU to host
        cudaMemcpyWrapper(host_biases[i].data(), accumulated_biases_gpu[i], 
                              biases_size * sizeof(double), cudaMemcpyDeviceToHost);
    }
}
