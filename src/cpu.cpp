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

// Sigmoid derivative for CPU
float sigmoid_derivative_cpu(float x) {
    // Apply sigmoid to get the output
    float sigmoid_output = sigmoid_cpu(x);
    // Derivative of sigmoid is sigmoid(x) * (1 - sigmoid(x))
    return sigmoid_output * (1.0f - sigmoid_output);
}

// Function for matrix multiplication
std::vector<float> matrix_multiply_cpu(const std::vector<float>& A, const std::vector<std::vector<float>>& B,
                                       const std::vector<float>& biases, int a_length_b_width, int b_length) {
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
        result[b_row] = sum + biases[b_row];
    }

    return result;
}

// Function to initialize memory, copy data, and run forward propagation
void forward_propagation_cpu(const std::vector<float>& input, std::vector<Layer>& network) {
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
        current_output = matrix_multiply_cpu(current_input, network[i].weights, network[i].biases, input_size, layer_output_size);

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

        network[i].output = current_output;
        // Prepare input for the next layer
        current_input = current_output;
        input_size = layer_output_size;
    }
}

float cost_cpu(const std::vector<float>& predicted, const std::vector<float>& actual) {
    float cost = 0.0f;
    for (size_t i = 0; i < predicted.size(); ++i) {
        cost += (predicted[i] - actual[i])*(predicted[i] - actual[i]);
    }

    return cost;
}

// // Function to calculate cross-entropy loss
// float cross_entropy_loss_cpu(const std::vector<float>& predicted, const std::vector<float>& actual) {
//     float loss = 0.0f;
//     for (size_t i = 0; i < predicted.size(); ++i) {
//         loss -= actual[i] * std::log(predicted[i] + 1e-15f); // Add small epsilon to avoid log(0)
//     }

//     return loss;
// }

// for now just change network after each training example
// later batch them together and calculate average. Then change.
// when working with "next" and "previous" keep in mind that we walk backwards here
// from "next" to "previous".
void backpropagation_cpu(const std::vector<float>& input, const std::vector<float>& actual, 
                         std::vector<Layer>& network, float learning_rate) {
    // Compute gradients and update weights and biases in reverse order
    std::vector<float> delta;

    std::cout << "Backpropagation:" << std::endl;
    for (int i = network.size() - 1; i >= 0; --i) {
        std::cout << "layer " << i << std::endl;
        const std::vector<float>& output = network[i].output;
        // previous output equal to derivatives of z in respect to all weights connected to this z
        const std::vector<float>& previous_output = (i == 0) ? input : network[i - 1].output;

        // Calculate the error
        // dC to dA(L), dA(L) to dZ are calculeted separately
        std::cout << "Computing deltas:" << GREY << std::endl;
        if (i == network.size() - 1) {
            // For output layer, use one hot vector (actual) to calculate dC to dA(L)
            delta.resize(output.size());
            // for each output
            for (size_t j = 0; j < output.size(); ++j) {
                // derivative of cost to a multiplied by derivative of a to z (sigmoid)
                // can remove 2.0f and reflect it in learning rate to remove this multiplication
                delta[j] = 2.0f * (output[j] - actual[j]) * sigmoid_derivative_cpu(output[j]);
                std::cout << "2.0f*(" << output[j] << "-" << actual[j] << ")*" << sigmoid_derivative_cpu(output[j]) << "=";
                std::cout << delta[j] << "\n";
            }
        } else {
            // For hidden layers, use previous(next) delta instead
            // sensitivity of z to the previous(next) activation is weight
            // this way we can calculate dC to dA(L)(in this case da(L-1)) for the next layer
            // so 2.0f * (output[j] - actual[j]) is replaced by a sum of next_delta and weights
            // 2.0f * (output[j] - actual[j]) * sigmoid_derivative_cpu(output[j]) * sum of weights for j
            std::vector<float> next_delta = delta;
            delta.resize(output.size());
            for (size_t j = 0; j < output.size(); ++j) {
                std::cout << "Delta " << j << ": (";
                delta[j] = 0.0f; // rewrite previous value
                for (size_t k = 0; k < next_delta.size(); ++k) {
                    // we are not in the last layer so i+1 is fine
                    delta[j] += next_delta[k] * network[i + 1].weights[k][j];
                    std::cout << next_delta[k] << "*" << network[i + 1].weights[k][j];
                    if (k > next_delta.size()-1) {
                        std::cout << " + ";
                    }
                }
                delta[j] *= sigmoid_derivative_cpu(output[j]);
                std::cout << ") * " << sigmoid_derivative_cpu(output[j]) << " = " << delta[j] << std::endl;
            }
        }
        std::cout << RESET << std::endl;

        std::cout << "Change weights and biases for layer " << i << ":"<< std::endl;
        // Update weights and biases
        for (size_t j = 0; j < network[i].weights.size(); ++j) {
            std::cout << "Weights " << j << ":" << GREY  << std::endl;
            for (size_t k = 0; k < network[i].weights[j].size(); ++k) {
                // dC/dw(L) = a(L-1)*delta
                std::cout << network[i].weights[j][k] << "-=" << learning_rate << "*" << delta[j] << "*" << previous_output[k] << " ";
                network[i].weights[j][k] -= learning_rate * delta[j] * previous_output[k];
            }
            std::cout << RESET << std::endl;
            // since z to bias derivative is 1 so dC/db(L) = 1*delta
            std::cout << "Biases:" << GREY << std::endl;
            std::cout << network[i].biases[j] << "-=" << learning_rate << "*" << delta[j] << " ";
            network[i].biases[j] -= learning_rate * delta[j];
            std::cout << RESET << std::endl;
        }
        std::cout << RESET << std::endl;
    }
}
