#include "network/train.h"

#include <algorithm>
#include <iomanip>
#include <thread>

#include "io/colors.h"
#include "io/io_utils.h"
#include "io/log.h"
#include "network/cpu.h"
#include "network/gpu.h"

void train_on_all_images_cpu(const float& learning_rate, const size_t& training_amount, const size_t& testing_amount, const size_t& batch_size,
                             const std::vector<uint8_t>& train_labels, const std::vector<uint8_t>& test_labels,
                             const std::vector<std::vector<float>>& train_images_norm,
                             const std::vector<std::vector<float>>& test_images_norm, std::vector<Layer> network) {
    auto start_time = std::chrono::high_resolution_clock::now();
    LOG(SL::NOTICE) << "(CPU) Training network with " << BLUE << training_amount << RESET << " images and " << BLUE
                    << learning_rate << RESET << " learning rate\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));  // Waits for 1 second
    LOG(SL::NOTICE) << std::endl;
    vec_float_2d image_batch(batch_size);
    std::vector<uint8_t> label_batch(batch_size);
    std::vector<size_t> counters(batch_size);
    for (size_t i = 0; i < training_amount; i++) {
        // everything will be fine
        image_batch[i % batch_size] = train_images_norm[i];
        label_batch[i % batch_size] = train_labels[i];
        counters[i % batch_size] = i;
        if (i % batch_size == batch_size-1) {
            train_on_a_batch_cpu(counters, learning_rate, label_batch, image_batch, network);
        }
    }
    auto train_end_time = std::chrono::high_resolution_clock::now();
    int correct_count = 0;

    LOG(SL::NOTICE) << "(CPU) Testing network with " << BLUE << testing_amount << RESET << " images\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));  // Waits for 1 second
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
                    << RESET << " for learing rate " << GREEN << learning_rate << RESET << " train duration: " << GREEN
                    << train_duration.count() << RESET << " test duration: " << GREEN << test_duration.count() << RESET
                    << std::endl;
}

void train_on_all_images_gpu(const float& learning_rate, size_t& training_amount, size_t& testing_amount,
                             const std::vector<uint8_t>& train_labels, const std::vector<uint8_t>& test_labels,
                             const std::vector<std::vector<float>>& train_images_norm,
                             const std::vector<std::vector<float>>& test_images_norm, std::vector<Layer> network) {
    auto start_time = std::chrono::high_resolution_clock::now();
    LOG(SL::NOTICE) << "(GPU) Training network with " << BLUE << training_amount << RESET << " images and " << BLUE
                    << learning_rate << RESET << " learning rate\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));  // Waits for 1 second
    LOG(SL::NOTICE) << std::endl;
    std::vector<GPULayer> gpu_network = allocate_network_data_on_gpu(network);
    for (size_t i = 0; i < training_amount; i++) {
        train_on_one_image_gpu(i, learning_rate, train_labels[i], train_images_norm[i], network, gpu_network);
        if (SL::should_log(SL::DEBUG)) {
            network_from_gpu_to_host(network, gpu_network);
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
                    << RESET << " for learing rate " << GREEN << learning_rate << RESET << " train duration: " << GREEN
                    << train_duration.count() << RESET << " test duration: " << GREEN << test_duration.count() << RESET
                    << std::endl;
}

void train_on_a_batch_cpu(const std::vector<size_t> counters, const float learning_rate, const std::vector<uint8_t>& labels,
                          const vec_float_2d& image_batch, std::vector<Layer>& network) {
    LOG(SL::NOTICE) << "(CPU) Processing images " << BLUE << static_cast<int>(counters[0]) << " - " << static_cast<int>(counters.back()) << RESET << std::endl;
    std::vector<float> one_hot_average(network.back().output.size(), 0.0f);
    vec_float_2d activations_average(network.size());
    std::vector<vec_float_2d> all_activations(image_batch.size());
    for (size_t j = 0; j < network.size(); j++) {
        activations_average[j] = std::vector<float>(network[j].output.size(), 0.0f);
    }
    // perform forward propagation for each image in a batch
    for (size_t i = 0; i < image_batch.size(); i++) {
        vec_float_2d activations = forward_propagation_cpu(image_batch[i], network);
        all_activations[i] = activations;
        for (size_t j = 0; j < activations_average.size(); j++) {
            for (size_t k = 0; k < activations_average[j].size(); k++) {
                activations_average[j][k] += activations[j][k];
            }
        }
        size_t label_index = labels[i];
        if (label_index < one_hot_average.size()) {
            one_hot_average[label_index] += 1.0f;
        }
    }
    for (size_t j = 0; j < activations_average.size(); j++) {
        for (size_t k = 0; k < activations_average[j].size(); k++) {
            activations_average[j][k] /= static_cast<float>(image_batch.size());
        }
        network[j].output = activations_average[j];
    }
    for (size_t j = 0; j < one_hot_average.size(); j++) {
        one_hot_average[j] /= static_cast<float>(image_batch.size());
    }

    float cumulative_loss = cumulative_loss_cpu(activations_average[network.size()-1], one_hot_average);
    LOG(SL::NOTICE) << "Cumulative Loss: " << BLUE << cumulative_loss << RESET << std::endl;

    // so here we pass all images and calculate gradient sum E(1-images.size())(delta * previous activation)
    backpropagation_cpu(image_batch, one_hot_average, network, learning_rate, all_activations);
    print_network(network);
}

void train_on_one_image_cpu(const int counter, const float learning_rate, const uint8_t& label,
                            const std::vector<float>& train_image, std::vector<Layer>& network) {
    print_image(train_image, 28, 28);
    LOG(SL::NOTICE) << "(CPU) Processing image " << BLUE << counter << RESET << ", label: " << BLUE
                    << static_cast<int>(label) << RESET << std::endl;
    std::vector<float> one_hot = generate_one_hot_vector(label);
    // Perform forward propagation on the GPU
    const vec_float_2d activations = forward_propagation_cpu(train_image, network);
    for (size_t i = 0; i < activations.size(); i++) {
        network[i].output = activations[i];
    }
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

    float cumulative_loss = cumulative_loss_cpu(output, one_hot);
    LOG(SL::NOTICE) << "Cumulative Loss: " << BLUE << cumulative_loss << RESET << std::endl;

    backpropagation_cpu({ train_image }, one_hot, network, learning_rate, { activations });
    print_network(network);
}

void train_on_one_image_gpu(const int counter, const float learning_rate, const uint8_t& label,
                            const std::vector<float>& train_image, std::vector<Layer>& network,
                            std::vector<GPULayer>& gpu_network) {
    print_image(train_image, 28, 28);
    LOG(SL::NOTICE) << "(GPU) Processing image " << BLUE << counter << RESET << ", label: " << BLUE
                    << static_cast<int>(label) << RESET << std::endl;
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

    float cumulative_loss = cumulative_loss_cpu(output, one_hot);
    LOG(SL::NOTICE) << "Cumulative Loss: " << BLUE << cumulative_loss << RESET << std::endl;

    backpropagation_gpu(train_image, one_hot, network, gpu_network, learning_rate);
    print_network(network);
}


bool test_one_image_cpu(const int counter, const uint8_t& label, const std::vector<float>& train_image,
                        std::vector<Layer>& network) {
    print_image(train_image, 28, 28);
    LOG(SL::NOTICE) << "(CPU) Processing image " << BLUE << counter << RESET << ", label: " << BLUE
                    << static_cast<int>(label) << RESET << std::endl;
    // Perform forward propagation on the GPU
    vec_float_2d activations = forward_propagation_cpu(train_image, network);
    for (size_t i = 0; i < activations.size(); i++) {
        network[i].output = activations[i];
    }
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
        LOG(SL::NOTICE) << GREEN << static_cast<int>(max_index) << RESET << " == " << GREEN << static_cast<int>(label)
                        << RESET << std::endl;
    } else {
        LOG(SL::NOTICE) << MAGENTA << static_cast<int>(max_index) << RESET << " == " << MAGENTA
                        << static_cast<int>(label) << RESET << std::endl;
    }

    return label == max_index;
}

bool test_one_image_gpu(const int counter, const uint8_t& label, const std::vector<float>& train_image,
                        std::vector<Layer>& network, std::vector<GPULayer>& gpu_network) {
    print_image(train_image, 28, 28);
    LOG(SL::NOTICE) << "(GPU) Processing image " << BLUE << counter << RESET << ", label: " << BLUE
                    << static_cast<int>(label) << RESET << std::endl;
    // Perform forward propagation on the GPU
    forward_propagation_gpu(train_image, network, gpu_network);
    print_network(network);
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
