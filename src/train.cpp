#include "train.h"

#include <algorithm>

#include "colors.h"
#include "log.h"
#include "cpu.h"
#include "gpu.h"
#include "io_utils.h"

void train_on_one_image_cpu(const int counter, const float learning_rate, const uint8_t& label, const std::vector<float>& train_image, std::vector<Layer>& network) {
    print_image(train_image, 28, 28);
    LOG(SL::NOTICE) << "(CPU) Processing image " << BLUE << counter << RESET << ", label: " << BLUE << static_cast<int>(label) << RESET << std::endl;
    std::vector<float> one_hot = generate_one_hot_vector(label);
    // Perform forward propagation on the GPU
    forward_propagation_cpu(train_image, network);
    // Print the output
    std::vector<float> output = network.back().output;
    LOG(SL::DEBUG) << "Output probabilities (CPU):" << GREY << std::endl;
    for (size_t i = 0; i < output.size(); ++i) {
        LOG(SL::DEBUG) << i << ": " << output[i];
        if (i < output.size() - 1) {
            LOG(SL::DEBUG) << ", ";
        }
    }
    LOG(SL::DEBUG) << RESET << std::endl;

    float cost = cost_cpu(output, one_hot);
    LOG(SL::NOTICE) << "Cost: " << BLUE << cost << RESET << std::endl;

    backpropagation_cpu(train_image, one_hot, network, learning_rate);
    print_network(network);
}

bool test_one_image_cpu(const int counter, const uint8_t& label, const std::vector<float>& train_image, std::vector<Layer>& network) {
    print_image(train_image, 28, 28);
    LOG(SL::NOTICE) << "(CPU) Processing image " << BLUE << counter << RESET << ", label: " << BLUE << static_cast<int>(label) << RESET << std::endl;
    // Perform forward propagation on the GPU
    forward_propagation_cpu(train_image, network);
    // Print the output
    std::vector<float>& output = network.back().output;
    LOG(SL::DEBUG) << "Output probabilities (CPU):" << GREY << std::endl;
    for (size_t i = 0; i < output.size(); ++i) {
        LOG(SL::DEBUG) << i << ": " << output[i];
        if (i < output.size() - 1) {
            LOG(SL::DEBUG) << ", ";
        }
    }
    LOG(SL::DEBUG) << RESET << std::endl;
    print_network(network);

    // Find the index of the highest probability
    uint8_t max_index = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    LOG(SL::NOTICE) << "Prediction: ";
    if (label == max_index) {
        LOG(SL::NOTICE) << GREEN << static_cast<int>(max_index) << RESET << " == " << GREEN << static_cast<int>(label) << RESET << std::endl;
    } else {
        LOG(SL::NOTICE) << MAGENTA << static_cast<int>(max_index) << RESET << " == " << MAGENTA << static_cast<int>(label) << RESET << std::endl;
    }

    return label == max_index;
}

void train_on_one_image_gpu(const int counter, const float learning_rate, const uint8_t& label, const std::vector<float>& train_image, std::vector<Layer>& network) {
    print_image(train_image, 28, 28);
    LOG(SL::NOTICE) << "(GPU) Processing image " << BLUE << counter << RESET << ", label: " << BLUE << static_cast<int>(label) << RESET << std::endl;
    std::vector<float> one_hot = generate_one_hot_vector(label);
    // Perform forward propagation on the GPU
    std::vector<GPULayer> gpu_network = forward_propagation_gpu(train_image, network);
    // Print the output
    std::vector<float> output = network.back().output;
    LOG(SL::DEBUG) << "Output probabilities (GPU):" << GREY << std::endl;
    for (size_t i = 0; i < output.size(); ++i) {
        LOG(SL::DEBUG) << i << ": " << output[i];
        if (i < output.size() - 1) {
            LOG(SL::DEBUG) << ", ";
        }
    }
    LOG(SL::DEBUG) << RESET << std::endl;

    float cost = cost_cpu(output, one_hot);
    LOG(SL::NOTICE) << "Cost: " << BLUE << cost << RESET << std::endl;

    backpropagation_gpu(train_image, one_hot, network, gpu_network, learning_rate);
    network_from_gpu_to_host(network, gpu_network);
    print_network(network);
}
