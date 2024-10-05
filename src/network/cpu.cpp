#include "network/cpu.h"

#include <cmath>
#include <cstdlib>  // For the exit function
#include <iomanip>
#include <iostream>
#include <vector>

#include "io/colors.h"
#include "io/log.h"
#include "network/init.h"

// Sigmoid function for CPU
double sigmoid_cpu(double x) {
    if (x >= 0) {
        return 1.0f / (1.0f + std::exp(-x));
    } else {
        double exp_x = std::exp(x);
        return exp_x / (1.0f + exp_x);
    }
}

// Function to apply Sigmoid activation to a vector
void apply_sigmoid_cpu(std::vector<double>& input) {
    for (double& val : input) {
        val = sigmoid_cpu(val);
    }
}

// Sigmoid derivative for CPU
double sigmoid_derivative_cpu(double x) {
    // Apply sigmoid to get the output
    double sigmoid_output = sigmoid_cpu(x);
    // Derivative of sigmoid is sigmoid(x) * (1 - sigmoid(x))
    return sigmoid_output * (1.0f - sigmoid_output);
}

double relu_derivative_cpu(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

// Function for matrix multiplication
std::vector<double> matrix_multiply_cpu(const std::vector<double>& A, const std::vector<std::vector<double>>& B,
                                       const std::vector<double>& biases, int a_length_b_width, int b_length) {
    std::vector<double> result(b_length);
    int a_length_size = A.size();
    int b_length_size = B.size();
    int b_width_size = B[0].size();
    if (a_length_size != b_width_size) {
        LOG(SL::ERROR) << "a_length != b_width, " << a_length_size << " != " << b_width_size << std::endl;
        exit(EXIT_FAILURE);
    }
    LOG(SL::TRACE) << "A length (size()): " << a_length_size << std::endl;
    LOG(SL::TRACE) << "B length (size()): " << b_length_size << std::endl;
    LOG(SL::TRACE) << "B width (size()): " << b_width_size << std::endl;
    LOG(SL::TRACE) << "a_length_b_width: " << a_length_b_width << std::endl;
    LOG(SL::TRACE) << "b_length: " << b_length << std::endl;
    for (int b_row = 0; b_row < b_length; ++b_row) {
        double sum = 0.0f;
        for (int ab_index = 0; ab_index < a_length_b_width; ++ab_index) {
            sum += A[ab_index] * B[b_row][ab_index];
        }
        result[b_row] = sum + biases[b_row];
    }

    return result;
}

// Function to initialize memory, copy data, and run forward propagation
vec_double_2d forward_propagation_cpu(const std::vector<double>& input, const std::vector<Layer>& network) {
    int input_size = input.size();
    int output_size = network.back().weights.size();
    vec_double_2d activations(network.size());

    // Allocate memory for the intermediate results
    std::vector<double> current_input = input;
    std::vector<double> current_output;

    LOG(SL::INFO) << "(CPU) Forward Propagation:" << std::endl;
    LOG(SL::DEBUG)
        << "------------------------------------ Activation Output (CPU) ------------------------------------"
        << std::endl;
    for (int i = 0; i < network.size(); ++i) {
        int layer_output_size = network[i].weights.size();

        // Perform matrix multiplication
        current_output =
            matrix_multiply_cpu(current_input, network[i].weights, network[i].biases, input_size, layer_output_size);

        LOG(SL::DEBUG) << "Layer " << i << " Pre-Activation Output (CPU):" << GREY << std::endl;
        // Debug: Check output after matrix multiplication
        for (int j = 0; j < layer_output_size; ++j) {
            LOG(SL::DEBUG) << current_output[j] << " ";
        }
        LOG(SL::DEBUG) << RESET << std::endl;

        // Apply Sigmoid activation function
        apply_sigmoid_cpu(current_output);

        LOG(SL::DEBUG) << "Layer " << i << " Post-Activation Output (CPU):" << GREY << std::endl;
        // Debug: Check output after activation
        for (int j = 0; j < layer_output_size; ++j) {
            LOG(SL::DEBUG) << current_output[j] << " ";
        }
        LOG(SL::DEBUG) << RESET << std::endl;

        activations[i] = current_output;
        // Prepare input for the next layer
        current_input = current_output;
        input_size = layer_output_size;
        LOG(SL::INFO) << DIM_GREEN << "layer " << i << RESET << std::endl;
    }

    return activations;
}

double cumulative_loss_cpu(const std::vector<double>& predicted, const std::vector<double>& actual) {
    double cumulative_loss = 0.0f;
    LOG(SL::DEBUG) << "Cumulative loss:" << GREY << std::endl;
    for (size_t i = 0; i < predicted.size(); ++i) {
        LOG(SL::DEBUG) << "(" << predicted[i] << " - " << actual[i] << ") * (" << predicted[i] << " - " << actual[i] << ")" << std::endl;
        cumulative_loss += (predicted[i] - actual[i]) * (predicted[i] - actual[i]);
    }
    LOG(SL::DEBUG) << RESET << std::endl;

    return cumulative_loss;
}

// for now just change network after each training example
// later batch them together and calculate average. Then change.
// when working with "next" and "previous" keep in mind that we walk backwards here
// from "next" to "previous".
void backpropagation_cpu(const vec_double_2d& image_batch, const std::vector<double>& actual,
                         std::vector<Layer>& network, std::vector<Layer> accumulated_network_changes,
                         const double learning_rate, const std::vector<vec_double_2d>& all_activations) {
    // Compute gradients and update weights and biases in reverse order
    vec_double_2d delta(network.size());

    LOG(SL::INFO) << "Backpropagation:" << std::endl;
    for (int i = network.size() - 1; i >= 0; --i) {
        const std::vector<double>& output = network[i].output;

        // Calculate the error
        // dC to dA(L), dA(L) to dZ are calculeted separately
        LOG(SL::DEBUG) << "Computing deltas for layer " << i << " (CPU)" << GREY << std::endl;
        if (i == network.size() - 1) {
            // For output layer, use one hot vector (actual) to calculate dC to dA(L)
            delta[i].resize(output.size());
            // for each output
            for (size_t j = 0; j < output.size(); ++j) {
                LOG(SL::DEBUG) << "Delta " << j << ": (";
                // derivative of loss to a multiplied by derivative of a to z (sigmoid)
                // can remove 2.0f and reflect it in learning rate to remove this multiplication
                delta[i][j] = (output[j] - actual[j]) * sigmoid_derivative_cpu(output[j]);
                // if (j == 0) {
                //     // printf("%f\n", delta[i][j]);
                //     printf("%f = (%f - %f) * %f\n", delta[i][j], output[j], actual[j], sigmoid_derivative_cpu(output[j]));
                // }
                LOG(SL::DEBUG) << std::fixed << std::setprecision(6) << output[j] << "-" << actual[j] << ") * "
                               << sigmoid_derivative_cpu(output[j]) << " = ";
                LOG(SL::DEBUG) << delta[i][j] << "\n";
            }
        } else {
            // For hidden layers, use previous(next) delta instead
            // sensitivity of z to the previous(next) activation is weight
            // this way we can calculate dC to dA(L)(in this case da(L-1)) for the next layer
            // so 2.0f * (output[j] - actual[j]) is replaced by a sum of next_delta and weights
            // 2.0f * (output[j] - actual[j]) * sigmoid_derivative_cpu(output[j]) * sum of weights for j
            std::vector<double>& next_delta = delta[i+1];
            delta[i].resize(output.size());
            for (size_t j = 0; j < output.size(); ++j) {
                LOG(SL::DEBUG) << "Delta " << j << ": (";
                delta[i][j] = 0.0f;  // rewrite previous value
                for (size_t k = 0; k < next_delta.size(); ++k) {
                    // we are not in the last layer so i+1 is fine
                    delta[i][j] += next_delta[k] * network[i + 1].weights[k][j];
                    LOG(SL::DEBUG) << next_delta[k] << "*" << network[i + 1].weights[k][j];
                    if (k < next_delta.size() - 1) {
                        LOG(SL::DEBUG) << " + ";
                    }
                }
                double sigmoid_derivative = relu_derivative_cpu(output[j]);
                // double sigmoid_derivative = sigmoid_derivative_cpu(output[j]);
                delta[i][j] *= sigmoid_derivative;
                LOG(SL::DEBUG) << ") * " << sigmoid_derivative << " = " << delta[i][j] << std::endl;
            }
        }
        LOG(SL::DEBUG) << RESET << std::endl;
    }

    size_t image_batch_size = image_batch.size();
    for (size_t l = 0; l < image_batch_size; l++) {
        for (int i = network.size() - 1; i >= 0; --i) {
            LOG(SL::DEBUG) << "Change weights and biases for layer " << i << ":" << std::endl;
            // previous output equal to derivatives of z in respect to all weights connected to this z
            const std::vector<double>& previous_output = (i == 0) ? image_batch[l] : all_activations[l][i - 1];
            // Update weights and biases
            for (size_t j = 0; j < network[i].weights.size(); ++j) {
                LOG(SL::DEBUG) << "Weights " << j << "x" << network[i].weights[j].size() << ":" << GREY << std::endl;
                for (size_t k = 0; k < previous_output.size(); ++k) {
                    // dC/dw(L) = a(L-1)*delta
                    LOG(SL::DEBUG) << accumulated_network_changes[i].weights[j][k] << "+=" << learning_rate << "*" << delta[i][j] << "*"
                                << previous_output[k] << " ";
                    accumulated_network_changes[i].weights[j][k] += learning_rate * delta[i][j] * previous_output[k];
                }
                LOG(SL::DEBUG) << RESET << std::endl;
                // since z to bias derivative is 1 so dC/db(L) = 1*delta
                LOG(SL::DEBUG) << "Biases:" << GREY << std::endl;
                LOG(SL::DEBUG) << accumulated_network_changes[i].biases[j] << "+=" << learning_rate << "*" << delta[i][j] << " ";
                accumulated_network_changes[i].biases[j] += learning_rate * delta[i][j];
                LOG(SL::DEBUG) << RESET << std::endl;
            }
            LOG(SL::DEBUG) << RESET << std::endl;
            LOG(SL::INFO) << DIM_GREEN << "layer " << i << RESET << std::endl;
        }
    }

    // for (size_t l = 0; l < image_batch_size; l++) {
    //     for (int i = network.size() - 1; i >= 2; --i) {
    //     // for (int i = network.size() - 2; i >= 1; --i) {
    //         const std::vector<double>& previous_output = (i == 0) ? image_batch[l] : all_activations[l][i - 1];
    //         // for (size_t j = 0; j < network[i].weights.size(); ++j) {
    //         for (size_t j = 0; j < 1; ++j) {
    //             LOG(SL::NOTICE) << "Weights " << j << "x" << network[i].weights.size() << ":" << GREY << std::endl;
    //             for (size_t k = 0; k < previous_output.size(); ++k) {
    //                 LOG(SL::NOTICE) << network[i].weights[j][k] << "-=" << learning_rate << "*" << delta[i][j] << "*"
    //                                 << previous_output[k] << " ";
    //                 // LOG(SL::NOTICE) << network[i].weights[j][k] << "-=" << accumulated_network_changes[i].weights[j][k] / image_batch_size << " ";
    //             }
    //             LOG(SL::NOTICE) << RESET << std::endl;
    //             LOG(SL::NOTICE) << "Biases:" << GREY << std::endl;
    //             LOG(SL::NOTICE) << network[i].biases[j] << "+=" << learning_rate << "*" << delta[i][j] << " ";
    //             // LOG(SL::NOTICE) << network[i].biases[j] << "-=" << accumulated_network_changes[i].biases[j] / image_batch_size << " ";
    //             LOG(SL::NOTICE) << RESET << std::endl;
    //         }
    //         LOG(SL::NOTICE) << RESET << std::endl;
    //     }
    // }

    // Apply the accumulated updates after all batches are processed
    for (int i = network.size() - 1; i >= 0; --i) {
        const std::vector<double>& previous_output = (i == 0) ? image_batch[0] : all_activations[0][i - 1];
        for (size_t j = 0; j < network[i].weights.size(); ++j) {
            for (size_t k = 0; k < previous_output.size(); ++k) {
                network[i].weights[j][k] -= accumulated_network_changes[i].weights[j][k] / image_batch_size;
            }
            network[i].biases[j] -= accumulated_network_changes[i].biases[j] / image_batch_size;
        }
    }
}
