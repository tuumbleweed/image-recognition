#ifndef TRAIN_H
#define TRAIN_H

#include <vector>

#include "network/init.h"
#include "network/gpu.h"

bool test_one_image_cpu(const int counter, const uint8_t& label, const std::vector<double>& train_image, std::vector<Layer>& network);

bool test_one_image_gpu(const int counter, const uint8_t& label, const std::vector<double>& train_image,
                        std::vector<Layer>& network, std::vector<GPULayer>& gpu_network);

void train_on_a_batch_cpu(const std::vector<size_t> counters, const double learning_rate, const std::vector<uint8_t>& labels,
                          const vec_double_2d& image_batch, std::vector<Layer>& network, std::vector<Layer> accumulated_network_changes);
void train_on_a_batch_gpu(const std::vector<size_t> counters, const double learning_rate, const std::vector<uint8_t>& labels,
                          const vec_double_2d& image_batch, std::vector<Layer>& network, std::vector<GPULayer>& gpu_network);

void train_on_all_images_cpu(const double& learning_rate, const size_t& training_amount, const size_t& testing_amount, const size_t& batch_size,
                             const std::vector<uint8_t>& train_labels, const std::vector<uint8_t>& test_labels,
                             const std::vector<std::vector<double>>& train_images_norm,
                             const std::vector<std::vector<double>>& test_images_norm,
                             std::vector<Layer> network, std::vector<Layer> accumulated_network_changes);


void train_on_all_images_gpu(const double& learning_rate, size_t& training_amount, size_t& testing_amount, const size_t& batch_size,
                             const std::vector<uint8_t>& train_labels, const std::vector<uint8_t>& test_labels,
                             const std::vector<std::vector<double>>& train_images_norm,
                             const std::vector<std::vector<double>>& test_images_norm, std::vector<Layer> network);

#endif // TRAIN_H