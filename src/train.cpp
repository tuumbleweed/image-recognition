#include "train.h"

#include "colors.h"
#include "log.h"
#include "cpu.h"

void train_on_one_image_cpu(uint8_t& label, std::vector<float>& train_image, std::vector<Layer>& network) {
    std::vector<float> one_hot = generate_one_hot_vector(label);
    // Perform forward propagation on the GPU
    forward_propagation_cpu(train_image, network);
    // Print the output
    std::vector<float> output = network[network.size() - 1].output;
    LOG(SL::DEBUG) << "Output probabilities (CPU):" << GREY << std::endl;
    for (size_t i = 0; i < output.size(); ++i) {
        LOG(SL::DEBUG) << i << ": " << output[i] << ", ";
    }
    LOG(SL::DEBUG) << RESET << std::endl;

    float cost = cost_cpu(output, one_hot);
    LOG(SL::INFO) << "Cost: " << BLUE << cost << RESET << std::endl;

    backpropagation_cpu(train_image, one_hot, network, 0.001);
    print_network(network);
}
