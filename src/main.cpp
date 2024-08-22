#include <cstdint>
#include <iostream>
#include <tuple>
#include <vector>

#include "colors.h"
#include "cpu.h"
#include "gpu.h"
#include "io_utils.h"
#include "layer.h"
#include "normalize.h"
#include "log.h"

std::vector<float> generate_one_hot_vector(uint8_t label, uint8_t length = 10) {
    std::vector<float> one_hot_vector(length, 0.0f);
    if (label < length) {
        one_hot_vector[label] = 1.0f;
    }
    return one_hot_vector;
}

int main() {
    SL::load_config("cfg/config.ini");

    auto [train_images, train_rows, train_cols] = read_images("tmp/train-images.idx3-ubyte");
    std::vector<uint8_t> train_labels = read_labels("tmp/train-labels.idx1-ubyte");
    auto [test_images, test_rows, test_cols] = read_images("tmp/t10k-images.idx3-ubyte");
    std::vector<uint8_t> test_labels = read_labels("tmp/t10k-labels.idx1-ubyte");

    std::vector<std::vector<float>> train_images_norm = normalize_all(train_images);
    std::vector<std::vector<float>> test_images_norm = normalize_all(test_images);

    int input_size = 28 * 28;  // 784 for 28x28 images
    int hidden_size1 = 16;
    int hidden_size2 = 16;
    int output_size = 10;  // 10 classes for digit recognition

    // Example: Using the first training image for forward propagation
    std::vector<float>& first_train_image = train_images_norm[0];
    uint8_t first_train_label = test_labels[0];
    // std::vector<float> output;

    std::vector<Layer> network = initialize_network(input_size, hidden_size1, hidden_size2, output_size);
    std::vector<float> one_hot = generate_one_hot_vector(first_train_label);

    // Perform forward propagation on the GPU
    forward_propagation_cpu(first_train_image, network);
    // Print the output
    std::vector<float> output = network[network.size() - 1].output;
    std::cout << "Output probabilities (CPU):" << GREY << std::endl;
    for (size_t i = 0; i < output.size(); ++i) {
        std::cout << i << ": " << output[i] << ", ";
    }
    std::cout << RESET << std::endl;

    float cost = cost_cpu(output, one_hot);
    std::cout << "Cost: " << BLUE << cost << RESET << std::endl;

    backpropagation_cpu(first_train_image, one_hot, network, 0.001);
    print_network(network);

    // std::cout << "Outputs (CPU):" << std::endl;
    // for (size_t i = 0; i < network.size(); ++i) {
    //     std::cout << "Output " << i << ":" << GREY << std::endl;
    //     for (size_t k = 0; k < network[i].output.size(); ++k) {
    //         std::cout << network[i].output[k] << " ";
    //     }
    //     std::cout << RESET << std::endl << std::endl;
    // }
    // std::cout << RESET << std::endl << std::endl;

    // forward_propagation_gpu(first_train_image, network, output);
    // // Print the output (softmax probabilities)
    // std::cout << "Output probabilities (GPU):" << GREY << std::endl;
    // for (size_t i = 0; i < output.size(); ++i) {
    //     std::cout << i << ": " << output[i] << ", ";
    // }
    // std::cout << RESET << std::endl << std::endl;

    return 0;
}
