#ifndef CPU_H
#define CPU_H

#include "network/init.h"

// Sigmoid function for CPU
double sigmoid_cpu(double x);

// Function to apply Sigmoid activation to a vector
void apply_sigmoid_cpu(std::vector<double>& input);

// Function for matrix multiplication
std::vector<double> matrix_multiply_cpu(const std::vector<double>& A, const std::vector<std::vector<double>>& B,
                                       const std::vector<double>& biases, int a_length_b_width, int b_length);

// Function to initialize memory, copy data, and run forward propagation
vec_double_2d forward_propagation_cpu(const std::vector<double>& input, const std::vector<Layer>& network);

double cumulative_loss_cpu(const std::vector<double>& predicted, const std::vector<double>& actual);

void backpropagation_cpu(const vec_double_2d& image_batch, const std::vector<double>& actual,
                         std::vector<Layer>& network, std::vector<Layer> accumulated_network_changes,
                         const double learning_rate, const std::vector<vec_double_2d>& all_activations);

#endif // CPU_H
