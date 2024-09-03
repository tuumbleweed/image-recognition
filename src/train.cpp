#include "train.h"

#include <algorithm>
#include <thread>

#include "io/colors.h"
#include "io/log.h"
#include "cpu.h"
#include "gpu.h"
#include "io/io_utils.h"
#include <iomanip>

void train_on_all_images_cpu(const float& learning_rate, size_t& training_amount, size_t& testing_amount,
                             const std::vector<uint8_t>& train_labels, const std::vector<uint8_t>& test_labels,
                             const std::vector<std::vector<float>>& train_images_norm,
                             const std::vector<std::vector<float>>& test_images_norm,
                             std::vector<Layer> network) {
    auto start_time = std::chrono::high_resolution_clock::now();
    LOG(SL::NOTICE) << "(CPU) Training network with " << BLUE << training_amount << RESET << " images and " << BLUE << learning_rate << RESET << " learning rate\n";
    std::this_thread::sleep_for(std::chrono::seconds(1)); // Waits for 1 second
    LOG(SL::NOTICE) << std::endl;
    for (size_t i=0; i<training_amount; i++) {
        train_on_one_image_cpu(i, learning_rate, train_labels[i], train_images_norm[i], network);
    }
    auto train_end_time = std::chrono::high_resolution_clock::now();

    int correct_count = 0;

    LOG(SL::NOTICE) << "(CPU) Testing network with " << BLUE << testing_amount << RESET << " images\n";
    std::this_thread::sleep_for(std::chrono::seconds(1)); // Waits for 1 second
    LOG(SL::NOTICE) << std::endl;
    for (int i = 0; i < testing_amount; ++i) {
        bool correct = test_one_image_cpu(i, test_labels[i], test_images_norm[i], network);
        if (correct) {
            ++correct_count;
        }
    }
    float success_rate = (static_cast<float>(correct_count) / testing_amount) * 100.0;
    auto test_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> train_duration = train_end_time - start_time;
    std::chrono::duration<double> test_duration = test_end_time - train_end_time;
    LOG(SL::NOTICE) << "(CPU) Got success rate " << GREEN << std::fixed << std::setprecision(2) << success_rate << "%"
        << RESET << " for learing rate " << GREEN << learning_rate << RESET
        << " train duration: " << GREEN << train_duration.count() << RESET
        << " test duration: " << GREEN << test_duration.count() << RESET << std::endl;
}

void train_on_all_images_gpu(const float& learning_rate, size_t& training_amount, size_t& testing_amount,
                             const std::vector<uint8_t>& train_labels, const std::vector<uint8_t>& test_labels,
                             const std::vector<std::vector<float>>& train_images_norm,
                             const std::vector<std::vector<float>>& test_images_norm,
                             std::vector<Layer> network) {

    auto start_time = std::chrono::high_resolution_clock::now();
    LOG(SL::NOTICE) << "(GPU) Training network with " << BLUE << training_amount << RESET << " images and " << BLUE << learning_rate << RESET << " learning rate\n";
    std::this_thread::sleep_for(std::chrono::seconds(1)); // Waits for 1 second
    LOG(SL::NOTICE) << std::endl;
    std::vector<GPULayer> gpu_network = allocate_network_data_on_gpu(network);
    for (size_t i=0; i<training_amount; i++) {
        train_on_one_image_gpu(i, learning_rate, train_labels[i], train_images_norm[i], network, gpu_network);
    }
    auto train_end_time = std::chrono::high_resolution_clock::now();

    int correct_count = 0;

    LOG(SL::NOTICE) << "(GPU) Testing network with " << BLUE << testing_amount << RESET << " images\n";
    std::this_thread::sleep_for(std::chrono::seconds(1)); // Waits for 1 second
    LOG(SL::NOTICE) << std::endl;
    for (int i = 0; i < testing_amount; ++i) {
        bool correct = test_one_image_gpu(i, test_labels[i], test_images_norm[i], network, gpu_network);
        if (correct) {
            ++correct_count;
        }
    }
    deallocate_network_data_on_gpu(gpu_network);
    float success_rate = (static_cast<float>(correct_count) / testing_amount) * 100.0;
    auto test_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> train_duration = train_end_time - start_time;
    std::chrono::duration<double> test_duration = test_end_time - train_end_time;
    LOG(SL::NOTICE) << "(GPU) Got success rate " << GREEN << std::fixed << std::setprecision(2) << success_rate << "%"
        << RESET << " for learing rate " << GREEN << learning_rate << RESET
        << " train duration: " << GREEN << train_duration.count() << RESET
        << " test duration: " << GREEN << test_duration.count() << RESET << std::endl;
}

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

void train_on_one_image_gpu(const int counter, const float learning_rate, const uint8_t& label, const std::vector<float>& train_image,
                            std::vector<Layer>& network, std::vector<GPULayer>& gpu_network) {
    print_image(train_image, 28, 28);
    LOG(SL::NOTICE) << "(GPU) Processing image " << BLUE << counter << RESET << ", label: " << BLUE << static_cast<int>(label) << RESET << std::endl;
    std::vector<float> one_hot = generate_one_hot_vector(label);
    // Perform forward propagation on the GPU
    forward_propagation_gpu(train_image, network, gpu_network);
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

bool test_one_image_gpu(const int counter, const uint8_t& label, const std::vector<float>& train_image,
                        std::vector<Layer>& network, std::vector<GPULayer>& gpu_network) {
    print_image(train_image, 28, 28);
    LOG(SL::NOTICE) << "(GPU) Processing image " << BLUE << counter << RESET << ", label: " << BLUE << static_cast<int>(label) << RESET << std::endl;
    // Perform forward propagation on the GPU
    forward_propagation_gpu(train_image, network, gpu_network);
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
    network_from_gpu_to_host(network, gpu_network);
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