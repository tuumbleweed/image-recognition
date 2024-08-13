#ifndef GPU_H
#define GPU_H

#include "layer.h"
#include <vector>

// Forward declaration of functions that are used in main.cpp
void forward_propagation_gpu(const std::vector<float>& input,
                             const std::vector<Layer>& network,
                             std::vector<float>& output);

#ifdef __CUDACC__ // This ensures the following code is only compiled by nvcc
__device__ double sigmoid_gpu(double x);
__global__ void apply_sigmoid_gpu(float* input, int size);
__global__ void matrix_multiply_gpu(const float* A, const float* B, float* result, int a_length_b_width, int b_length);
#endif

#endif // GPU_H
