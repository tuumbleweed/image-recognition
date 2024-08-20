#ifndef LAYER_H
#define LAYER_H

#include <vector>

// Structure to represent a layer in the neural network
struct Layer {
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;
    std::vector<float> output;

    // Constructor to initialize weights and biases with given lengths
    Layer(size_t input_dim, size_t output_dim) {
        weights = std::vector<std::vector<float>>(output_dim, std::vector<float>(input_dim, 0.0f));
        biases = std::vector<float>(output_dim, 0.0f);
        output = std::vector<float>(output_dim, 0.0f);
    }
};

// Helper function to initialize the weights of a layer
std::vector<std::vector<float>> initialize_weights(int input_dim, int output_dim);

// Function to initialize the network layers and their weights
std::vector<Layer> initialize_network(int input_size, int hidden_size1, int hidden_size2, int output_size);

#endif // LAYER_H
