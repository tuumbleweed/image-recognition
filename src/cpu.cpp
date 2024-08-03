#include "layer.h"
#include "cpu.h"
#include "colors.h"
#include <iostream>
#include <vector>
#include <cmath>

// Sigmoid function for CPU
float sigmoid_cpu(float x) {
    if (x >= 0) {
        return 1.0f / (1.0f + std::exp(-x));
    } else {
        float exp_x = std::exp(x);
        return exp_x / (1.0f + exp_x);
    }
}

// Function to apply Sigmoid activation to a vector
void apply_sigmoid_cpu(std::vector<float>& input) {
    for (auto& val : input) {
        val = sigmoid_cpu(val);
    }
}

// Function for matrix multiplication
void matrix_multiply_cpu(const std::vector<float>& A, const std::vector<std::vector<float>>& B, std::vector<float>& C, int M, int N, int K) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < K; ++col) {
            float value = 0.0f;
            for (int e = 0; e < N; ++e) {
                value += A[row * N + e] * B[col][e];
            }
            C[row * K + col] = value;
        }
    }
}

// Function to initialize memory, copy data, and run forward propagation
void forward_propagation_cpu(const std::vector<float>& input,
                             const std::vector<Layer>& network,
                             std::vector<float>& output) {
    int input_size = input.size();
    int output_size = network.back().weights.size();

    // Allocate memory for the intermediate results
    std::vector<float> current_input = input;
    std::vector<float> current_output;

    for (size_t i = 0; i < network.size(); ++i) {
        int layer_output_size = network[i].weights.size();
        current_output.resize(layer_output_size);

        // Perform matrix multiplication
        matrix_multiply_cpu(current_input, network[i].weights, current_output, 1, input_size, layer_output_size);

        std::cout << "Layer " << i << " Pre-Activation Output (CPU):" << GREY << std::endl;
        // Debug: Check output after matrix multiplication
        for (int j = 0; j < layer_output_size; ++j) {
            std::cout << current_output[j] << " ";
        }
        std::cout << RESET << std::endl;

        // Apply Sigmoid activation function
        apply_sigmoid_cpu(current_output);

        std::cout << "Layer " << i << " Post-Activation Output (CPU):" << GREY << std::endl;
        // Debug: Check output after activation
        for (int j = 0; j < layer_output_size; ++j) {
            std::cout << current_output[j] << " ";
        }
        std::cout << RESET << std::endl;

        // Prepare input for the next layer
        current_input = current_output;
        input_size = layer_output_size;
    }

    // Set the final output
    output = current_output;
}

