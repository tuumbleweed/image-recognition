#ifndef LAYER_H
#define LAYER_H

#include <vector>

// Structure to represent a layer in the neural network
struct Layer {
    std::vector<std::vector<float>> weights;
};

// Helper function to initialize the weights of a layer
std::vector<std::vector<float>> initialize_weights(int input_dim, int output_dim);

// Function to initialize the network layers and their weights
std::vector<Layer> initialize_network(int input_size, int hidden_size1, int hidden_size2, int output_size);

#endif // LAYER_H
