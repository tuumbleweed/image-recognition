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

    int input_size = 28 * 28;  // 784 for 28x28 images
    int hidden_size1 = 16;
    int hidden_size2 = 16;
    int output_size = 10;  // 10 classes for digit recognition
    std::vector<Layer> network = initialize_network(input_size, hidden_size1, hidden_size2, output_size);

    size_t how_many = vm["number"].as<size_t>();
    if (how_many == -1) {
        how_many = train_images_norm.size();
    }
    for (size_t i=0; i<how_many; i++) {
        train_on_one_image_cpu(i, train_labels[i], train_images_norm[i], network);
    }

    // forward_propagation_gpu(first_train_image, network, output);
    // // Print the output (softmax probabilities)
    // LOG(SL::DEBUG) << "Output probabilities (GPU):" << GREY << std::endl;
    // for (size_t i = 0; i < output.size(); ++i) {
    //     LOG(SL::DEBUG) << i << ": " << output[i] << ", ";
    // }
    // LOG(SL::DEBUG) << RESET << std::endl << std::endl;

    return 0;
}
