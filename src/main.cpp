#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>
#include <tuple>
#include <vector>

#include "io/cml.h"
#include "io/colors.h"
#include "io/io_utils.h"
#include "io/log.h"
#include "io/normalize.h"
#include "network/cpu.h"
#include "network/gpu.h"
#include "network/init.h"
#include "network/train.h"

int main(int argc, char *argv[]) {
    po::variables_map vm = parse_command_line(argc, argv);
    SL::set_log_level(SL::string_to_log_level(vm["log"].as<std::string>()));
    unsigned int seed = vm["seed"].as<unsigned int>();

    const auto [train_images, train_rows, train_cols] = read_images("tmp/train-images.idx3-ubyte");
    const std::vector<uint8_t> train_labels = read_labels("tmp/train-labels.idx1-ubyte");
    const auto [test_images, test_rows, test_cols] = read_images("tmp/t10k-images.idx3-ubyte");
    const std::vector<uint8_t> test_labels = read_labels("tmp/t10k-labels.idx1-ubyte");

    const std::vector<std::vector<double>> train_images_norm = normalize_all(train_images);
    const std::vector<std::vector<double>> test_images_norm = normalize_all(test_images);

    const int input_size = train_rows * train_cols;  // 784 for 28x28 images
    // 128x32, 1.29 lr 8 per batch = 82.44%
    // 512x256, 1.4 lr 3 per batch = 90.17%
    const int hidden_size1 = 512;
    const int hidden_size2 = 256;
    const int output_size = 10;  // 10 classes for digit recognition
    const double learning_rate = vm["learning-rate"].as<double>();
    const bool cpu = vm["cpu"].as<bool>();
    const bool gpu = vm["gpu"].as<bool>();
    std::vector<Layer> network = initialize_network(input_size, hidden_size1, hidden_size2, output_size, seed);
    std::vector<Layer> network_copy = network;
    const std::vector<Layer> accumulated_network_changes = initialize_network(input_size, hidden_size1, hidden_size2, output_size, seed, true);

    size_t training_amount = vm["train"].as<size_t>();
    if (training_amount == -1 || (training_amount > train_images_norm.size())) {
        training_amount = train_images_norm.size();
    }
    size_t testing_amount = vm["test"].as<size_t>();
    if ((testing_amount == -1) || (testing_amount > test_images_norm.size())) {
        testing_amount = test_images_norm.size();
    }
    size_t batch_size = vm["batch-size"].as<size_t>();

    if (cpu) {
        train_on_all_images_cpu(learning_rate, training_amount, testing_amount, batch_size, train_labels, test_labels,
                                train_images_norm, test_images_norm, network, accumulated_network_changes);
    }

    if (gpu) {
        train_on_all_images_gpu(learning_rate, training_amount, testing_amount, batch_size, train_labels, test_labels,
                                train_images_norm, test_images_norm, network_copy);
    }

    return 0;
}
