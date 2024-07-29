#include "io_utils.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <tuple>

int main() {
    auto [train_images, train_rows, train_cols] = read_images("tmp/train-images.idx3-ubyte");
    std::vector<uint8_t> train_labels = read_labels("tmp/train-labels.idx1-ubyte");
    auto [test_images, test_rows, test_cols] = read_images("tmp/t10k-images.idx3-ubyte");
    std::vector<uint8_t> test_labels = read_labels("tmp/t10k-labels.idx1-ubyte");

    std::cout << "Label " << static_cast<int>(train_labels[0]) << std::endl;
    std::vector<uint8_t> first_train_image = train_images[0];
    print_image(first_train_image, train_rows, train_cols);

    std::cout << "Label " << static_cast<int>(test_labels[0]) << std::endl;
    std::vector<uint8_t> first_test_image = test_images[0];
    print_image(first_test_image, test_rows, test_cols);

    return 0;
}
