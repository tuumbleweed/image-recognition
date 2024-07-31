#include "layer.h"
#include <random>
#include <cmath>
#include <iostream>

// Helper function to initialize the weights of a layer
std::vector<std::vector<float>> initialize_weights(int input_dim, int output_dim) {
    std::vector<std::vector<float>> layer_weights(output_dim, std::vector<float>(input_dim));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(input_dim)); // Xavier initialization

    for (int i = 0; i < output_dim; ++i) {
        for (int j = 0; j < input_dim; ++j) {
            layer_weights[i][j] = dist(gen);
        }
    }
    return layer_weights;
}

// Function to initialize the network layers and their weights
std::vector<Layer> initialize_network(int input_size, int hidden_size1, int hidden_size2, int output_size) {
    std::cout << "Initializing network " << input_size << 'x' << hidden_size1 << 'x' << hidden_size2 << 'x' << output_size << std::endl;
    std::vector<Layer> network;

    // Initialize weights between input layer and first hidden layer
    Layer input_to_hidden1 = {initialize_weights(input_size, hidden_size1)};
    network.push_back(input_to_hidden1);

    // Initialize weights between first hidden layer and second hidden layer
    Layer hidden1_to_hidden2 = {initialize_weights(hidden_size1, hidden_size2)};
    network.push_back(hidden1_to_hidden2);

    // Initialize weights between second hidden layer and output layer
    Layer hidden2_to_output = {initialize_weights(hidden_size2, output_size)};
    network.push_back(hidden2_to_output);

    return network;
}
