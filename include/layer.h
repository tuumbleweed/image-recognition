#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <cstdint>

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

Layer initialize_weights_and_biases(const int input_dim, const int output_dim, unsigned int seed);

// Function to initialize the network layers and their weights
std::vector<Layer> initialize_network(int input_size, int hidden_size1, int hidden_size2, int output_size, unsigned int seed);

std::vector<float> generate_one_hot_vector(uint8_t label, uint8_t length = 10);

#endif // LAYER_H
