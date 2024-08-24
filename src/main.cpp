#include <iostream>
#include <tuple>
#include <vector>
#include <iomanip>
#include <thread>
#include <chrono>

#include "colors.h"
#include "cpu.h"
#include "gpu.h"
#include "io_utils.h"
#include "layer.h"
#include "normalize.h"
#include "log.h"
#include "train.h"
#include "cml.h"


int main(int argc, char *argv[]) {
    po::variables_map vm = parse_command_line(argc, argv);
    SL::load_config(vm["config"].as<std::string>());

    const auto [train_images, train_rows, train_cols] = read_images("tmp/train-images.idx3-ubyte");
    const std::vector<uint8_t> train_labels = read_labels("tmp/train-labels.idx1-ubyte");
    const auto [test_images, test_rows, test_cols] = read_images("tmp/t10k-images.idx3-ubyte");
    const std::vector<uint8_t> test_labels = read_labels("tmp/t10k-labels.idx1-ubyte");

    const std::vector<std::vector<float>> train_images_norm = normalize_all(train_images);
    const std::vector<std::vector<float>> test_images_norm = normalize_all(test_images);

    const int input_size = train_rows * train_cols;  // 784 for 28x28 images
    const int hidden_size1 = 16;
    const int hidden_size2 = 16;
    const int output_size = 10;  // 10 classes for digit recognition
    const float learning_rate = vm["learning-rate"].as<float>();
    std::vector<Layer> network = initialize_network(input_size, hidden_size1, hidden_size2, output_size);

    size_t how_many = vm["number"].as<size_t>();
    if (how_many == -1) {
        how_many = train_images_norm.size();
    }
    int total_tests = test_labels.size();

    auto cpu_start_time = std::chrono::high_resolution_clock::now();
    LOG(SL::NOTICE) << "(CPU) Training network with " << BLUE << how_many << RESET << " images and " << BLUE << learning_rate << RESET << " learning rate\n";
    std::this_thread::sleep_for(std::chrono::seconds(1)); // Waits for 1 second
    LOG(SL::NOTICE) << std::endl;
    for (size_t i=0; i<how_many; i++) {
        train_on_one_image_cpu(i, learning_rate, train_labels[i], train_images_norm[i], network);
    }
    auto cpu_train_end_time = std::chrono::high_resolution_clock::now();

    int correct_count = 0;

    for (int i = 0; i < total_tests; ++i) {
        bool correct = test_one_image_cpu(i, test_labels[i], test_images_norm[i], network);
        if (correct) {
            ++correct_count;
        }
    }
    float cpu_success_rate = (static_cast<float>(correct_count) / total_tests) * 100.0;
    auto cpu_test_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_train_duration = cpu_train_end_time - cpu_start_time;
    std::chrono::duration<double> cpu_test_duration = cpu_test_end_time - cpu_train_end_time;
    LOG(SL::NOTICE) << "(CPU) Got success rate " << GREEN << std::fixed << std::setprecision(2) << cpu_success_rate << "%"
        << RESET << " for learing rate " << GREEN << learning_rate << RESET
        << " train duration: " << GREEN << cpu_train_duration.count() << RESET
        << " test duration: " << GREEN << cpu_test_duration.count() << RESET << std::endl;

    auto gpu_start_time = std::chrono::high_resolution_clock::now();
    LOG(SL::NOTICE) << "(GPU) Training network with " << BLUE << how_many << RESET << " images and " << BLUE << learning_rate << RESET << " learning rate\n";
    std::this_thread::sleep_for(std::chrono::seconds(1)); // Waits for 1 second
    LOG(SL::NOTICE) << std::endl;
    for (size_t i=0; i<how_many; i++) {
        train_on_one_image_gpu(i, learning_rate, train_labels[i], train_images_norm[i], network);
    }
    auto gpu_train_end_time = std::chrono::high_resolution_clock::now();

    int gpu_correct_count = 0;

    for (int i = 0; i < total_tests; ++i) {
        bool correct = test_one_image_cpu(i, test_labels[i], test_images_norm[i], network);
        if (correct) {
            ++correct_count;
        }
    }
    float gpu_success_rate = (static_cast<float>(correct_count) / total_tests) * 100.0;
    auto gpu_test_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_train_duration = gpu_train_end_time - gpu_start_time;
    std::chrono::duration<double> gpu_test_duration = gpu_test_end_time - gpu_train_end_time;
    LOG(SL::NOTICE) << "(GPU) Got success rate " << GREEN << std::fixed << std::setprecision(2) << gpu_success_rate << "%"
        << RESET << " for learing rate " << GREEN << learning_rate << RESET
        << " train duration: " << GREEN << gpu_train_duration.count() << RESET
        << " test duration: " << GREEN << gpu_test_duration.count() << RESET << std::endl;

    // forward_propagation_gpu(train_images_norm[0], network);
    // LOG(SL::DEBUG) << "Output probabilities (GPU):" << GREY << std::endl;
    // for (size_t i = 0; i < network[network.size()-1].output.size(); ++i) {
    //     LOG(SL::DEBUG) << i << ": " << network[network.size()-1].output[i] << ", ";
    // }
    // LOG(SL::DEBUG) << RESET << std::endl << std::endl;

    return 0;
}
