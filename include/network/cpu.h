#ifndef CPU_H
#define CPU_H

#include "network/init.h"

// Sigmoid function for CPU
float sigmoid_cpu(float x);

// Function to apply Sigmoid activation to a vector
void apply_sigmoid_cpu(std::vector<float>& input);

// Function for matrix multiplication
std::vector<float> matrix_multiply_cpu(const std::vector<float>& A, const std::vector<std::vector<float>>& B,
                                       const std::vector<float>& biases, int a_length_b_width, int b_length);

// Function to initialize memory, copy data, and run forward propagation
vec_float_2d forward_propagation_cpu(const std::vector<float>& input, const std::vector<Layer>& network);

float cumulative_loss_cpu(const std::vector<float>& predicted, const std::vector<float>& actual);

void backpropagation_cpu(const vec_float_2d& image_batch, const std::vector<float>& actual, std::vector<Layer>& network,
                         const float learning_rate, const std::vector<vec_float_2d>& all_activations);

#endif // CPU_H
