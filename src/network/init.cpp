#include "network/init.h"

#include <cmath>
#include <iostream>
#include <random>
#include <thread>

#include "io/colors.h"
#include "io/log.h"

// Function to initialize the network layers and their weights
std::vector<Layer> initialize_network(const int input_size, const int hidden_size1, const int hidden_size2,
                                      const int output_size, unsigned int seed, bool zero_init) {
    LOG(SL::NOTICE) << "Initializing network " << BLUE << input_size << 'x' << hidden_size1;
    LOG(SL::NOTICE) << 'x' << hidden_size2 << 'x' << output_size << RESET;
    LOG(SL::NOTICE) << " with seed " << BLUE << seed << RESET;
    if (seed == 0) {
        LOG(SL::NOTICE) << BLUE << "(random)" << RESET;
    }
    LOG(SL::NOTICE) << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(1));  // Waits for 1 second

    std::vector<Layer> network;

    // Initialize weights between input layer and first hidden layer
    Layer input_to_hidden1 = initialize_weights_and_biases(input_size, hidden_size1, seed, zero_init);
    network.push_back(input_to_hidden1);

    // Initialize weights between first hidden layer and second hidden layer
    Layer hidden1_to_hidden2 = initialize_weights_and_biases(hidden_size1, hidden_size2, seed, zero_init);
    network.push_back(hidden1_to_hidden2);

    // Initialize weights between second hidden layer and output layer
    Layer hidden2_to_output = initialize_weights_and_biases(hidden_size2, output_size, seed, zero_init);
    network.push_back(hidden2_to_output);

    std::this_thread::sleep_for(std::chrono::seconds(1));  // Waits for 1 second
    LOG(SL::NOTICE) << std::endl;

    return network;
}

// Helper function to initialize the weights of a layer
Layer initialize_weights_and_biases(const int input_dim, const int output_dim, unsigned int seed, bool zero_init) {
    Layer layer(input_dim, output_dim);
    std::random_device rd;
    std::mt19937 gen(seed);
    if (seed == 0) {
        gen.seed(rd());
    }
    std::normal_distribution<double> dist(0.0f, 1.0f / std::sqrt(input_dim));  // Xavier initialization

    LOG(SL::DEBUG) << "------------------------------------ Initialize weights (" << input_dim << 'x' << output_dim
                   << ") ------------------------------------" << std::endl;
    for (int i = 0; i < output_dim; ++i) {
        LOG(SL::DEBUG) << "Initialize weights (" << input_dim << 'x' << output_dim << "), " << i + 1 << '/'
                       << output_dim << ":" << GREY << std::endl;
        for (int j = 0; j < input_dim; ++j) {
            if (zero_init) {
                layer.weights[i][j] = 0.0f;
            } else {
                layer.weights[i][j] = dist(gen);
            }
            LOG(SL::DEBUG) << layer.weights[i][j] << " ";
        }
        LOG(SL::DEBUG) << RESET << std::endl;
    }
    LOG(SL::DEBUG) << "Initialize Biases (" << 1 << 'x' << output_dim << "):" << GREY << std::endl;
    // 2 loops only because I want to print stuff separately
    for (int i = 0; i < output_dim; ++i) {
        if (zero_init) {
            layer.biases[i] = 0.0f;
        } else {
            layer.biases[i] = dist(gen);
        }
        LOG(SL::DEBUG) << layer.biases[i] << " ";
    }
    LOG(SL::DEBUG) << RESET << "\n" << std::endl;

    return layer;
}

std::vector<double> generate_one_hot_vector(uint8_t label, uint8_t length) {
    std::vector<double> one_hot_vector(length, 0.0f);
    if (label < length) {
        one_hot_vector[label] = 1.0f;
    }
    if (SL::should_log(SL::DEBUG)) {
        LOG(SL::DEBUG) << "One hot: " << GREY;
        for (double& value : one_hot_vector) {
            LOG(SL::DEBUG) << value << " ";
        }
        LOG(SL::DEBUG) << RESET << std::endl;
    }
    return one_hot_vector;
}
