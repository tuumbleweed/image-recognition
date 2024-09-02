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
    SL::set_log_level(SL::string_to_log_level(vm["log"].as<std::string>()));

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
    const bool cpu = vm["cpu"].as<bool>();
    const bool gpu = vm["gpu"].as<bool>();
    std::vector<Layer> network = initialize_network(input_size, hidden_size1, hidden_size2, output_size);
    std::vector<Layer> network_copy = network;

    size_t how_many = vm["number"].as<size_t>();
    if (how_many == -1) {
        how_many = train_images_norm.size();
    }
    size_t total_tests = vm["test"].as<size_t>();
    if ((total_tests == -1) || (total_tests > test_images_norm.size())) {
        total_tests = test_images_norm.size();
    }

    if (cpu) {
        auto start_time = std::chrono::high_resolution_clock::now();
        LOG(SL::NOTICE) << "(CPU) Training network with " << BLUE << how_many << RESET << " images and " << BLUE << learning_rate << RESET << " learning rate\n";
        std::this_thread::sleep_for(std::chrono::seconds(1)); // Waits for 1 second
        LOG(SL::NOTICE) << std::endl;
        for (size_t i=0; i<how_many; i++) {
            train_on_one_image_cpu(i, learning_rate, train_labels[i], train_images_norm[i], network);
        }
        auto train_end_time = std::chrono::high_resolution_clock::now();

        int correct_count = 0;

        LOG(SL::NOTICE) << "(CPU) Testing network with " << BLUE << total_tests << RESET << " images\n";
        std::this_thread::sleep_for(std::chrono::seconds(1)); // Waits for 1 second
        LOG(SL::NOTICE) << std::endl;
        for (int i = 0; i < total_tests; ++i) {
            bool correct = test_one_image_cpu(i, test_labels[i], test_images_norm[i], network);
            if (correct) {
                ++correct_count;
            }
        }
        float success_rate = (static_cast<float>(correct_count) / total_tests) * 100.0;
        auto test_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> train_duration = train_end_time - start_time;
        std::chrono::duration<double> test_duration = test_end_time - train_end_time;
        LOG(SL::NOTICE) << "(CPU) Got success rate " << GREEN << std::fixed << std::setprecision(2) << success_rate << "%"
            << RESET << " for learing rate " << GREEN << learning_rate << RESET
            << " train duration: " << GREEN << train_duration.count() << RESET
            << " test duration: " << GREEN << test_duration.count() << RESET << std::endl;
    }

    if (gpu) {
        auto start_time = std::chrono::high_resolution_clock::now();
        LOG(SL::NOTICE) << "(GPU) Training network with " << BLUE << how_many << RESET << " images and " << BLUE << learning_rate << RESET << " learning rate\n";
        std::this_thread::sleep_for(std::chrono::seconds(1)); // Waits for 1 second
        LOG(SL::NOTICE) << std::endl;
        for (size_t i=0; i<how_many; i++) {
            train_on_one_image_gpu(i, learning_rate, train_labels[i], train_images_norm[i], network_copy);
        }
        auto train_end_time = std::chrono::high_resolution_clock::now();

        int correct_count = 0;

        LOG(SL::NOTICE) << "(GPU) Testing network with " << BLUE << total_tests << RESET << " images\n";
        std::this_thread::sleep_for(std::chrono::seconds(1)); // Waits for 1 second
        LOG(SL::NOTICE) << std::endl;
        for (int i = 0; i < total_tests; ++i) {
            bool correct = test_one_image_gpu(i, test_labels[i], test_images_norm[i], network_copy);
            if (correct) {
                ++correct_count;
            }
        }
        float success_rate = (static_cast<float>(correct_count) / total_tests) * 100.0;
        auto test_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> train_duration = train_end_time - start_time;
        std::chrono::duration<double> test_duration = test_end_time - train_end_time;
        LOG(SL::NOTICE) << "(GPU) Got success rate " << GREEN << std::fixed << std::setprecision(2) << success_rate << "%"
            << RESET << " for learing rate " << GREEN << learning_rate << RESET
            << " train duration: " << GREEN << train_duration.count() << RESET
            << " test duration: " << GREEN << test_duration.count() << RESET << std::endl;
    }

    return 0;
}
