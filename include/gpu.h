#ifndef GPU_H
#define GPU_H

#include "layer.h"
#include <vector>

struct GPULayer {
    float* weights; // flattened
    float* biases;
    float* output;
};

// Forward declaration of functions that are used in main.cpp
void forward_propagation_gpu(const std::vector<float>& input, std::vector<Layer>& network, std::vector<GPULayer>& gpu_network);
void backpropagation_gpu(const std::vector<float>& input, const std::vector<float>& actual, 
                         std::vector<Layer>& network, std::vector<GPULayer>& gpu_network, float learning_rate);
void network_from_gpu_to_host(std::vector<Layer>& network, std::vector<GPULayer>& gpu_network);
std::vector<GPULayer> allocate_network_data_on_gpu(std::vector<Layer>& network);
void deallocate_network_data_on_gpu(std::vector<GPULayer>& gpu_network);

#ifdef __CUDACC__ // This ensures the following code is only compiled by nvcc
__device__ float sigmoid_gpu(float x);
__global__ void apply_sigmoid_gpu(float* input, int size);
__global__ void matrix_multiply_gpu(const float* A, const float* B, const float* biases, float* result, int a_length_b_width, int b_length);
__device__ float sigmoid_derivative_gpu(float x);
#endif

#endif // GPU_H
