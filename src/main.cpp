#include "io_utils.h"
#include "normalize.h"
#include "layer.h"
#include "gpu.h"
#include "cpu.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <tuple>

void print_network(const std::vector<Layer>& network) {
    for (size_t layer_index = 0; layer_index < network.size(); ++layer_index) {
        std::cout << "Layer " << layer_index << " weights:" << std::endl;
        const auto& layer = network[layer_index];
        for (size_t i = 0; i < layer.weights.size(); ++i) {
            for (size_t j = 0; j < layer.weights[i].size(); ++j) {
                std::cout << layer.weights[i][j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

int main() {
    auto [train_images, train_rows, train_cols] = read_images("tmp/train-images.idx3-ubyte");
    std::vector<uint8_t> train_labels = read_labels("tmp/train-labels.idx1-ubyte");
    auto [test_images, test_rows, test_cols] = read_images("tmp/t10k-images.idx3-ubyte");
    std::vector<uint8_t> test_labels = read_labels("tmp/t10k-labels.idx1-ubyte");

    std::vector<std::vector<float>> train_images_norm = normalize_all(train_images);
    std::vector<std::vector<float>> test_images_norm = normalize_all(test_images);

    int input_size = 28 * 28; // 784 for 28x28 images
    int hidden_size1 = 16;
    int hidden_size2 = 16;
    int output_size = 10; // 10 classes for digit recognition

    std::vector<Layer> network = initialize_network(input_size, hidden_size1, hidden_size2, output_size);
    // print_network(network);

    // Example: Using the first training image for forward propagation
    std::vector<float>& first_train_image = train_images_norm[0];
    std::vector<float> output;

    // Perform forward propagation on the GPU
    forward_propagation_cpu(first_train_image, network, output);
    // Print the output (softmax probabilities)
    std::cout << "Output probabilities:" << std::endl;
    for (size_t i = 0; i < output.size(); ++i) {
        std::cout << "Class " << i << ": " << output[i] << std::endl;
    }

    forward_propagation_gpu(first_train_image, network, output);
    // Print the output (softmax probabilities)
    std::cout << "Output probabilities:" << std::endl;
    for (size_t i = 0; i < output.size(); ++i) {
        std::cout << "Class " << i << ": " << output[i] << std::endl;
    }

    // forward_propagation_gpu()
    // std::cout << "Label " << static_cast<int>(train_labels[0]) << std::endl;
    // std::vector<float> first_train_image = train_images_norm[0];
    // print_image(first_train_image, train_rows, train_cols);

    // std::cout << "Label " << static_cast<int>(test_labels[0]) << std::endl;
    // std::vector<float> first_test_image = test_images_norm[0];
    // print_image(first_test_image, test_rows, test_cols);

    return 0;
}
