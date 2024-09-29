#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <cstdint>
#include <cstddef> // for size_t (it was indirectlly included in previous versions but not in 12.2)

// Structure to represent a layer in the neural network
struct Layer {
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    std::vector<double> output;

    // Constructor to initialize weights and biases with given lengths
    Layer(size_t input_dim, size_t output_dim) {
        weights = std::vector<std::vector<double>>(output_dim, std::vector<double>(input_dim, 0.0f));
        biases = std::vector<double>(output_dim, 0.0f);
        output = std::vector<double>(output_dim, 0.0f);
    }
};

using vec_double_2d = std::vector<std::vector<double>>;

Layer initialize_weights_and_biases(const int input_dim, const int output_dim, unsigned int seed, bool zero_init);

// Function to initialize the network layers and their weights
std::vector<Layer> initialize_network(int input_size, int hidden_size1, int hidden_size2, int output_size, unsigned int seed, bool zero_init = false);

std::vector<double> generate_one_hot_vector(uint8_t label, uint8_t length = 10);

#endif // LAYER_H
