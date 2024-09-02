#ifndef TRAIN_H
#define TRAIN_H

#include <vector>

#include "layer.h"
#include "gpu.h"

void train_on_one_image_cpu(const int counter, const float learning_rate, const uint8_t& label, const std::vector<float>& train_image, std::vector<Layer>& network);

void train_on_one_image_gpu(const int counter, const float learning_rate, const uint8_t& label, const std::vector<float>& train_image,
                            std::vector<Layer>& network, std::vector<GPULayer>& gpu_network);

bool test_one_image_cpu(const int counter, const uint8_t& label, const std::vector<float>& train_image, std::vector<Layer>& network);

bool test_one_image_gpu(const int counter, const uint8_t& label, const std::vector<float>& train_image,
                        std::vector<Layer>& network, std::vector<GPULayer>& gpu_network);

void train_on_all_images_cpu(const float& learning_rate, size_t& training_amount, size_t& testing_amount,
                             const std::vector<uint8_t>& train_labels, const std::vector<uint8_t>& test_labels,
                             const std::vector<std::vector<float>>& train_images_norm,
                             const std::vector<std::vector<float>>& test_images_norm,
                             std::vector<Layer> network);


void train_on_all_images_gpu(const float& learning_rate, size_t& training_amount, size_t& testing_amount,
                             const std::vector<uint8_t>& train_labels, const std::vector<uint8_t>& test_labels,
                             const std::vector<std::vector<float>>& train_images_norm,
                             const std::vector<std::vector<float>>& test_images_norm,
                             std::vector<Layer> network);

#endif // TRAIN_H