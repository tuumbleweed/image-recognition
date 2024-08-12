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
std::vector<float> matrix_multiply_cpu(const std::vector<float>& A, const std::vector<std::vector<float>>& B, int b_length, int b_width) {
    std::vector<float> result(b_width);
    for (int b_col = 0; b_col < b_width; ++b_col) {
        float sum = 0.0f;
        for (int a_index = 0; a_index < b_length; ++a_index) {
            sum += A[a_index] * B[a_index][b_col];
        }
        result[b_col] = sum;
    }

    return result;
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

    std::cout << "------------------------------------Activation Output (CPU)------------------------------------" << std::endl;
    for (size_t i = 0; i < network.size(); ++i) {
        int layer_output_size = network[i].weights.size();

        // Perform matrix multiplication
        current_output = matrix_multiply_cpu(current_input, network[i].weights, input_size, layer_output_size);

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
