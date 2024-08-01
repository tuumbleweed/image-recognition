#ifndef CPU_H
#define CPU_H

#include "layer.h"

// Sigmoid function for CPU
float sigmoid_cpu(float x);

// Function to apply Sigmoid activation to a vector
void apply_sigmoid_cpu(std::vector<float>& input);

// Function for matrix multiplication
void matrix_multiply_cpu(const std::vector<float>& A, const std::vector<std::vector<float>>& B, std::vector<float>& C, int M, int N, int K);

// Function to initialize memory, copy data, and run forward propagation
void forward_propagation_cpu(const std::vector<float>& input,
                             const std::vector<Layer>& network,
                             std::vector<float>& output);


#endif // CPU_H
