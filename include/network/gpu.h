#ifndef GPU_H
#define GPU_H

#include "network/init.h"
#include <vector>

struct GPULayer {
    double* weights; // flattened
    double* biases;
    double* output;
};

// Forward declaration of functions that are used in main.cpp
vec_double_2d forward_propagation_gpu(const std::vector<double>& input, const std::vector<Layer>& network,
                             std::vector<GPULayer>& gpu_network);
void backpropagation_gpu(const vec_double_2d& image_batch, const std::vector<double>& actual, std::vector<Layer>& network,
                         std::vector<GPULayer>& gpu_network, const double learning_rate, const std::vector<vec_double_2d>& all_activations);
void network_from_gpu_to_host(std::vector<Layer>& network, std::vector<GPULayer>& gpu_network);
std::vector<GPULayer> allocate_network_data_on_gpu(std::vector<Layer>& network);
void deallocate_network_data_on_gpu(std::vector<GPULayer>& gpu_network);
void copy_accumulated_weights_to_host(
    const std::vector<double*>& delta,
    const std::vector<double*>& accumulated_weights_gpu,
    const std::vector<double*>& accumulated_biases_gpu,
    std::vector<std::vector<double>>& host_delta,
    std::vector<std::vector<double>>& host_weights,
    std::vector<std::vector<double>>& host_biases,
    const std::vector<Layer>& network
);

#ifdef __CUDACC__ // This ensures the following code is only compiled by nvcc
__device__ double sigmoid_gpu(double x);
__global__ void apply_sigmoid_gpu(double* input, int size);
__global__ void matrix_multiply_gpu(const double* A, const double* B, const double* biases, double* result, int a_length_b_width, int b_length);
__device__ double sigmoid_derivative_gpu(double x);
#endif

#endif // GPU_H
