#include "cpu.h"

#include <cmath>
#include <iostream>
#include <vector>
#include <cstdlib> // For the exit function

#include "colors.h"
#include "layer.h"

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
std::vector<float> matrix_multiply_cpu(const std::vector<float>& A, const std::vector<std::vector<float>>& B,
                                       int a_length_b_width, int b_length) {
    std::vector<float> result(b_length);
    int a_length_size = A.size();
    int b_length_size = B.size();
    int b_width_size = B[0].size();
    if (a_length_size != b_width_size) {
        std::cerr << "a_length != b_width, " << a_length_size << " != " << b_width_size << std::endl;
        exit(EXIT_FAILURE);
    }
    // std::cout << "A length (size()): " << a_length_size << std::endl;
    // std::cout << "B length (size()): " << b_length_size << std::endl;
    // std::cout << "B width (size()): " << b_width_size << std::endl;
    // std::cout << "a_length_b_width: " << a_length_b_width << std::endl;
    // std::cout << "b_length: " << b_length << std::endl;
    for (int b_row = 0; b_row < b_length; ++b_row) {
        float sum = 0.0f;
        for (int ab_index = 0; ab_index < a_length_b_width; ++ab_index) {
            sum += A[ab_index] * B[b_row][ab_index];
        }
        result[b_row] = sum;
    }

    return result;
}

// Function to initialize memory, copy data, and run forward propagation
void forward_propagation_cpu(const std::vector<float>& input, const std::vector<Layer>& network,
                             std::vector<float>& output) {
    int input_size = input.size();
    int output_size = network.back().weights.size();

    // Allocate memory for the intermediate results
    std::vector<float> current_input = input;
    std::vector<float> current_output;

    std::cout << "------------------------------------Activation Output (CPU)------------------------------------"
              << std::endl;
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
