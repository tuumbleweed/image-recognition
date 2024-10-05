#include "network/train.h"

#include <algorithm>
#include <iomanip>
#include <thread>

#include "io/colors.h"
#include "io/io_utils.h"
#include "io/log.h"
#include "network/cpu.h"
#include "network/gpu.h"

void train_on_all_images_cpu(const double& learning_rate, const size_t& training_amount, const size_t& testing_amount, const size_t& batch_size,
                             const std::vector<uint8_t>& train_labels, const std::vector<uint8_t>& test_labels,
                             const std::vector<std::vector<double>>& train_images_norm,
                             const std::vector<std::vector<double>>& test_images_norm,
                             std::vector<Layer> network, const std::vector<Layer> accumulated_network_changes) {
    auto start_time = std::chrono::high_resolution_clock::now();
    LOG(SL::NOTICE) << "(CPU) Training network with " << BLUE << training_amount << RESET << " images and " << BLUE
                    << learning_rate << RESET << " learning rate\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));  // Waits for 1 second
    LOG(SL::NOTICE) << std::endl;
    vec_double_2d image_batch(batch_size);
    std::vector<uint8_t> label_batch(batch_size);
    std::vector<size_t> counters(batch_size);
    for (size_t i = 0; i < training_amount; i++) {
        // everything will be fine
        size_t counter = i % batch_size;
        image_batch[counter] = train_images_norm[i];
        label_batch[counter] = train_labels[i];
        counters[counter] = i+1;
        if ((counter == batch_size-1) || (i == training_amount-1)) {
            if ((i == training_amount-1) && (counter != batch_size-1)) {
                image_batch.resize(counter+1);
                label_batch.resize(counter+1);
                counters.resize(counter+1);
            }
            train_on_a_batch_cpu(counters, learning_rate, label_batch, image_batch, network, accumulated_network_changes);
            // print_network(network);
        }
    }
    auto train_end_time = std::chrono::high_resolution_clock::now();
    int correct_count = 0;
    print_network(network);

    LOG(SL::NOTICE) << "(CPU) Testing network with " << BLUE << testing_amount << RESET << " images\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));  // Waits for 1 second
    LOG(SL::NOTICE) << std::endl;
    for (int i = 0; i < testing_amount; ++i) {
        double success_rate = (static_cast<double>(correct_count) / (i+1)) * 100.0;
        LOG(SL::NOTICE) << "(GPU) Got success rate " << GREEN << std::fixed << std::setprecision(2) << success_rate << "%" << RESET << std::endl;
        bool correct = test_one_image_cpu(i, test_labels[i], test_images_norm[i], network);
        if (correct) {
            ++correct_count;
        }
    }
    double success_rate = (static_cast<double>(correct_count) / testing_amount) * 100.0;
    auto test_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> train_duration = train_end_time - start_time;
    std::chrono::duration<double> test_duration = test_end_time - train_end_time;
    LOG(SL::NOTICE) << "(CPU) Got success rate " << GREEN << std::fixed << std::setprecision(2) << success_rate << "%"
                    << RESET << " for learing rate " << GREEN << learning_rate << RESET << " batch size "
                    << GREEN << batch_size << RESET << " train duration: " << GREEN << train_duration.count()
                    << RESET << " test duration: " << GREEN << test_duration.count() << RESET
                    << std::endl;
}

void train_on_all_images_gpu(const double& learning_rate, size_t& training_amount, size_t& testing_amount, const size_t& batch_size,
                             const std::vector<uint8_t>& train_labels, const std::vector<uint8_t>& test_labels,
                             const std::vector<std::vector<double>>& train_images_norm,
                             const std::vector<std::vector<double>>& test_images_norm, std::vector<Layer> network) {
    auto start_time = std::chrono::high_resolution_clock::now();
    LOG(SL::NOTICE) << "(GPU) Training network with " << BLUE << training_amount << RESET << " images and " << BLUE
                    << learning_rate << RESET << " learning rate\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));  // Waits for 1 second
    LOG(SL::NOTICE) << std::endl;
    std::vector<GPULayer> gpu_network = allocate_network_data_on_gpu(network);
    vec_double_2d image_batch(batch_size);
    std::vector<uint8_t> label_batch(batch_size);
    std::vector<size_t> counters(batch_size);
    for (size_t i = 0; i < training_amount; i++) {
        // everything will be fine
        size_t counter = i % batch_size;
        image_batch[counter] = train_images_norm[i];
        label_batch[counter] = train_labels[i];
        counters[counter] = i+1;
        if ((counter == batch_size-1) || (i == training_amount-1)) {
            if ((i == training_amount-1) && (counter != batch_size-1)) {
                image_batch.resize(counter+1);
                label_batch.resize(counter+1);
                counters.resize(counter+1);
            }
            train_on_a_batch_gpu(counters, learning_rate, label_batch, image_batch, network, gpu_network);
            // network_from_gpu_to_host(network, gpu_network);
            // print_network(network);
        }
    }
    auto train_end_time = std::chrono::high_resolution_clock::now();
    network_from_gpu_to_host(network, gpu_network);
    print_network(network);
    int correct_count = 0;

    LOG(SL::NOTICE) << "(GPU) Testing network with " << BLUE << testing_amount << RESET << " images\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));  // Waits for 1 second
    LOG(SL::NOTICE) << std::endl;
    for (int i = 0; i < testing_amount; ++i) {
        double success_rate = (static_cast<double>(correct_count) / (i+1)) * 100.0;
        LOG(SL::NOTICE) << "(GPU) Got success rate " << GREEN << std::fixed << std::setprecision(2) << success_rate << "%" << RESET << std::endl;
        bool correct = test_one_image_gpu(i, test_labels[i], test_images_norm[i], network, gpu_network);
        if (correct) {
            ++correct_count;
        }
    }
    deallocate_network_data_on_gpu(gpu_network);
    double success_rate = (static_cast<double>(correct_count) / testing_amount) * 100.0;
    auto test_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> train_duration = train_end_time - start_time;
    std::chrono::duration<double> test_duration = test_end_time - train_end_time;
    LOG(SL::NOTICE) << "(GPU) Got success rate " << GREEN << std::fixed << std::setprecision(2) << success_rate << "%"
                    << RESET << " for learing rate " << GREEN << learning_rate << RESET << " batch size "
                    << GREEN << batch_size << RESET << " train duration: " << GREEN << train_duration.count()
                    << RESET << " test duration: " << GREEN << test_duration.count() << RESET
                    << std::endl;
}

void train_on_a_batch_cpu(const std::vector<size_t> counters, const double learning_rate, const std::vector<uint8_t>& labels,
                          const vec_double_2d& image_batch, std::vector<Layer>& network, const std::vector<Layer> accumulated_network_changes) {
    LOG(SL::NOTICE) << "(CPU) Processing images " << BLUE << static_cast<int>(counters[0]) << " - " << static_cast<int>(counters.back()) << RESET << std::endl;
    std::vector<double> one_hot_average(network.back().output.size(), 0.0f);
    vec_double_2d activations_average(network.size());
    std::vector<vec_double_2d> all_activations(image_batch.size());
    for (size_t j = 0; j < network.size(); j++) {
        activations_average[j] = std::vector<double>(network[j].output.size(), 0.0f);
    }
    // perform forward propagation for each image in a batch
    for (size_t i = 0; i < image_batch.size(); i++) {
        all_activations[i] = forward_propagation_cpu(image_batch[i], network);
        for (size_t j = 0; j < activations_average.size(); j++) {
            for (size_t k = 0; k < activations_average[j].size(); k++) {
                activations_average[j][k] += all_activations[i][j][k];
            }
        }
        size_t label_index = labels[i];
        if (label_index < one_hot_average.size()) {
            one_hot_average[label_index] += 1.0f;
        }
    }
    for (size_t j = 0; j < activations_average.size(); j++) {
        for (size_t k = 0; k < activations_average[j].size(); k++) {
            activations_average[j][k] /= static_cast<double>(image_batch.size());
        }
        network[j].output = activations_average[j];
    }
    for (size_t j = 0; j < one_hot_average.size(); j++) {
        one_hot_average[j] /= static_cast<double>(image_batch.size());
    }

    double cumulative_loss = cumulative_loss_cpu(activations_average[network.size()-1], one_hot_average);
    LOG(SL::NOTICE) << "Cumulative Loss: " << BLUE << cumulative_loss << RESET << std::endl;

    // so here we pass all images and calculate gradient sum E(1-images.size())(delta * previous activation)
    backpropagation_cpu(image_batch, one_hot_average, network, accumulated_network_changes, learning_rate, all_activations);
}

void train_on_a_batch_gpu(const std::vector<size_t> counters, const double learning_rate, const std::vector<uint8_t>& labels,
                          const vec_double_2d& image_batch, std::vector<Layer>& network, std::vector<GPULayer>& gpu_network) {
    LOG(SL::NOTICE) << "(GPU) Processing images " << BLUE << static_cast<int>(counters[0]) << " - " << static_cast<int>(counters.back()) << RESET << std::endl;
    std::vector<double> one_hot_average(network.back().output.size(), 0.0f);
    vec_double_2d activations_average(network.size());
    std::vector<vec_double_2d> all_activations(image_batch.size());
    for (size_t j = 0; j < network.size(); j++) {
        activations_average[j] = std::vector<double>(network[j].output.size(), 0.0f);
    }
    // perform forward propagation for each image in a batch
    for (size_t i = 0; i < image_batch.size(); i++) {
        all_activations[i] = forward_propagation_gpu(image_batch[i], network, gpu_network);
        for (size_t j = 0; j < activations_average.size(); j++) {
            for (size_t k = 0; k < activations_average[j].size(); k++) {
                activations_average[j][k] += all_activations[i][j][k];
            }
        }
        size_t label_index = labels[i];
        if (label_index < one_hot_average.size()) {
            one_hot_average[label_index] += 1.0f;
        }
    }
    for (size_t j = 0; j < activations_average.size(); j++) {
        for (size_t k = 0; k < activations_average[j].size(); k++) {
            activations_average[j][k] /= static_cast<double>(image_batch.size());
        }
        network[j].output = activations_average[j];
    }
    for (size_t j = 0; j < one_hot_average.size(); j++) {
        one_hot_average[j] /= static_cast<double>(image_batch.size());
    }

    double cumulative_loss = cumulative_loss_cpu(activations_average[network.size()-1], one_hot_average);
    LOG(SL::NOTICE) << "Cumulative Loss: " << BLUE << cumulative_loss << RESET << std::endl;

    // so here we pass all images and calculate gradient sum E(1-images.size())(delta * previous activation)
    backpropagation_gpu(image_batch, one_hot_average, network, gpu_network, learning_rate, all_activations);
}

bool test_one_image_cpu(const int counter, const uint8_t& label, const std::vector<double>& train_image,
                        std::vector<Layer>& network) {
    print_image(train_image, 28, 28);
    LOG(SL::NOTICE) << "(CPU) Processing image " << BLUE << counter << RESET << ", label: " << BLUE
                    << static_cast<int>(label) << RESET << std::endl;
    // Perform forward propagation on the GPU
    vec_double_2d activations = forward_propagation_cpu(train_image, network);
    for (size_t i = 0; i < activations.size(); i++) {
        network[i].output = activations[i];
    }
    print_network(network);
    // Print the output
    std::vector<double>& output = network.back().output;
    LOG(SL::DEBUG) << "Output probabilities (CPU):" << GREY << std::endl;
    for (size_t i = 0; i < output.size(); ++i) {
        LOG(SL::DEBUG) << i << ": " << output[i];
        if (i < output.size() - 1) {
            LOG(SL::DEBUG) << ", ";
        }
    }
    LOG(SL::DEBUG) << RESET << std::endl;

    // Find the index of the highest probability
    uint8_t max_index = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    LOG(SL::NOTICE) << "Prediction: ";
    if (label == max_index) {
        LOG(SL::NOTICE) << GREEN << static_cast<int>(max_index) << RESET << " == " << GREEN << static_cast<int>(label)
                        << RESET << std::endl;
    } else {
        LOG(SL::NOTICE) << MAGENTA << static_cast<int>(max_index) << RESET << " == " << MAGENTA
                        << static_cast<int>(label) << RESET << std::endl;
    }

    return label == max_index;
}

bool test_one_image_gpu(const int counter, const uint8_t& label, const std::vector<double>& train_image,
                        std::vector<Layer>& network, std::vector<GPULayer>& gpu_network) {
    print_image(train_image, 28, 28);
    LOG(SL::NOTICE) << "(GPU) Processing image " << BLUE << counter << RESET << ", label: " << BLUE
                    << static_cast<int>(label) << RESET << std::endl;
    // Perform forward propagation on the GPU
    // forward_propagation_gpu2(train_image, network, gpu_network);
    vec_double_2d activations = forward_propagation_gpu(train_image, network, gpu_network);
    for (size_t i = 0; i < activations.size(); i++) {
        network[i].output = activations[i];
    }
    print_network(network);
    // Print the output
    std::vector<double>& output = network.back().output;
    LOG(SL::DEBUG) << "Output probabilities (CPU):" << GREY << std::endl;
    for (size_t i = 0; i < output.size(); ++i) {
        LOG(SL::DEBUG) << i << ": " << output[i];
        if (i < output.size() - 1) {
            LOG(SL::DEBUG) << ", ";
        }
    }
    LOG(SL::DEBUG) << RESET << std::endl;

    // Find the index of the highest probability
    uint8_t max_index = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    LOG(SL::NOTICE) << "Prediction: ";
    if (label == max_index) {
        LOG(SL::NOTICE) << GREEN << static_cast<int>(max_index) << RESET << " == " << GREEN << static_cast<int>(label)
                        << RESET << std::endl;
    } else {
        LOG(SL::NOTICE) << MAGENTA << static_cast<int>(max_index) << RESET << " == " << MAGENTA
                        << static_cast<int>(label) << RESET << std::endl;
    }

    return label == max_index;
}
