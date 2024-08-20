#include "layer.h"

#include <cmath>
#include <iostream>
#include <random>

#include "colors.h"

// Helper function to initialize the weights of a layer
Layer initialize_weights_and_biases(int input_dim, int output_dim) {
    // std::vector<std::vector<float>> layer_weights(output_dim, std::vector<float>(input_dim));
    Layer layer(input_dim, output_dim);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(input_dim));  // Xavier initialization

    std::cout << "------------------------------------Initialize weights (" << input_dim << 'x' << output_dim
              << ")------------------------------------" << std::endl;
    for (int i = 0; i < output_dim; ++i) {
        std::cout << "Initialize weights (" << input_dim << 'x' << output_dim << "), " << i + 1 << '/' << output_dim
                  << ":" << GREY << std::endl;
        for (int j = 0; j < input_dim; ++j) {
            layer.weights[i][j] = dist(gen);
            std::cout << layer.weights[i][j] << " ";
        }
        std::cout << RESET << std::endl;
        layer.biases[i] = dist(gen);
    }
    std::cout << "Initialize Biases (" << 1 << 'x' << output_dim << "):" << GREY << std::endl;
    for (int i = 0; i < output_dim; ++i) {
        std::cout << layer.biases[i] << " ";
    }
    std::cout << RESET << "\n" << std::endl;
    return layer;
}

// Function to initialize the network layers and their weights
std::vector<Layer> initialize_network(int input_size, int hidden_size1, int hidden_size2, int output_size) {
    std::cout << "Initializing network " << input_size << 'x' << hidden_size1;
    std::cout << 'x' << hidden_size2 << 'x' << output_size << std::endl;
    std::vector<Layer> network;

    // Initialize weights between input layer and first hidden layer
    Layer input_to_hidden1 = initialize_weights_and_biases(input_size, hidden_size1);
    network.push_back(input_to_hidden1);

    // Initialize weights between first hidden layer and second hidden layer
    Layer hidden1_to_hidden2 = initialize_weights_and_biases(hidden_size1, hidden_size2);
    network.push_back(hidden1_to_hidden2);

    // Initialize weights between second hidden layer and output layer
    Layer hidden2_to_output = initialize_weights_and_biases(hidden_size2, output_size);
    network.push_back(hidden2_to_output);

    return network;
}

void print_network(const std::vector<Layer>& network) {
    for (int k = 0; k < network.size(); ++k) {
        const Layer& layer = network[k];
        std::cout << "------------------------------------Layer " << k+1 << "------------------------------------" << std::endl;
        for (int i = 0; i < layer.weights.size(); ++i) {
            std::cout << "Weights " << i + 1 << '/' << layer.weights.size()
                    << ":" << GREY << std::endl;
            for (const float& weight : layer.weights[i]) {
                 std::cout << weight << " ";
            }
            std::cout << RESET << std::endl;
        }
        std::cout << "Biases (" << 1 << 'x' << layer.biases.size() << "):" << GREY << std::endl;
        for (int i = 0; i < layer.biases.size(); ++i) {
            std::cout << layer.biases[i] << " ";
        }
        std::cout << RESET << "\n" << std::endl;
        std::cout << "Activations (" << 1 << 'x' << layer.output.size() << "):" << GREY << std::endl;
        for (int i = 0; i < layer.output.size(); ++i) {
            std::cout << layer.output[i] << " ";
        }
        std::cout << RESET << "\n" << std::endl;
    }
}
