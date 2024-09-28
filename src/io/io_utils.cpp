#include "io/io_utils.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <tuple>

#include "io/colors.h"
#include "io/log.h"

std::tuple<std::vector<std::vector<uint8_t>>, int, int> read_images(const std::string& filename) {
    LOG(SL::NOTICE) << "Reading images " << BLUE << filename << RESET << std::endl;
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        LOG(SL::ERROR) << "Cannot open file: " << filename << std::endl;
        return {};
    }

    int32_t magic_number = 0;
    int32_t number_of_images = 0;
    int32_t rows = 0;
    int32_t cols = 0;

    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char*>(&number_of_images), sizeof(number_of_images));
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    magic_number = __builtin_bswap32(magic_number);
    number_of_images = __builtin_bswap32(number_of_images);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    std::vector<std::vector<uint8_t>> images(number_of_images, std::vector<uint8_t>(rows * cols));

    for (int i = 0; i < number_of_images; ++i) {
        file.read(reinterpret_cast<char*>(images[i].data()), rows * cols);
    }

    file.close();
    return std::make_tuple(images, rows, cols);
}

std::vector<uint8_t> read_labels(const std::string& filename) {
    LOG(SL::NOTICE) << "Reading labels " << BLUE << filename << RESET << std::endl;
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        LOG(SL::ERROR) << "Cannot open file: " << filename << std::endl;
        return {};
    }

    int32_t magic_number = 0;
    int32_t number_of_labels = 0;

    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char*>(&number_of_labels), sizeof(number_of_labels));

    magic_number = __builtin_bswap32(magic_number);
    number_of_labels = __builtin_bswap32(number_of_labels);

    std::vector<uint8_t> labels(number_of_labels);
    file.read(reinterpret_cast<char*>(labels.data()), labels.size());

    file.close();
    return labels;
}

// print image as grayscale ascii art
void print_image(const std::vector<double>& image, const int& rows, const int& cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (image[r * cols + c] == 0.0f) {
                LOG(SL::DEBUG) << std::setw(5) << 0 << " ";
            } else {
                LOG(SL::DEBUG) << std::fixed << std::setprecision(3) << std::setw(5) << image[r * cols + c] << " ";
            }
        }
        LOG(SL::DEBUG) << std::endl;
    }
}

// print every weight, bias and activation (output)
void print_network(const std::vector<Layer>& network) {
    LOG(SL::INFO) << "Printing network with " << network.size() << " layers: " << BLUE;
    for (int k = 0; k < network.size(); ++k) {
        LOG(SL::INFO) << "(" << network[k].weights.size() << "x" << network[k].weights[0].size() << ")";
        if (k < network.size() - 1) {
            LOG(SL::INFO) << "x";
        }
    }
    LOG(SL::INFO) << RESET << std::endl;
    for (int k = 0; k < network.size(); ++k) {
        const Layer& layer = network[k];
        LOG(SL::DEBUG) << "------------------------------------ Layer " << k << " ------------------------------------"
                       << std::endl;
        for (int i = 0; i < layer.weights.size(); ++i) {
            LOG(SL::DEBUG) << "Weights " << i + 1 << '/' << layer.weights.size() << ":" << GREY << std::endl;
            for (const double& weight : layer.weights[i]) {
                LOG(SL::DEBUG) << weight << " ";
            }
            LOG(SL::DEBUG) << RESET << std::endl;
        }
        LOG(SL::DEBUG) << "Biases (" << 1 << 'x' << layer.biases.size() << "):" << GREY << std::endl;
        for (int i = 0; i < layer.biases.size(); ++i) {
            LOG(SL::DEBUG) << layer.biases[i] << " ";
        }
        LOG(SL::DEBUG) << RESET << "\n" << std::endl;
        SL::LogLevel log_level = SL::INFO;
        if (k == network.size() - 1) {
            log_level = SL::NOTICE;
        }
        LOG(log_level) << "Activations (" << 1 << 'x' << layer.output.size() << "): " << GREY;
        for (int i = 0; i < layer.output.size(); ++i) {
            LOG(log_level) << layer.output[i] << " ";
        }
        LOG(log_level) << RESET << std::endl;
    }
    LOG(SL::NOTICE) << std::endl;
}
