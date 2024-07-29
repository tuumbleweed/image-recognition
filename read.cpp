#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>  // for std::setw
#include <tuple>

void print_image(std::vector<uint8_t>& image, int& rows, int& cols) {
    // Print the whole image
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            std::cout << std::setw(3) << static_cast<int>(image[r * cols + c]) << " ";
        }
        std::cout << std::endl;
    }
}

std::tuple<std::vector<std::vector<uint8_t>>, int, int> read_images(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
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
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
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

    // std::cout << train_images.size() << std::endl;
    // std::cout << train_labels.size() << std::endl;
    // std::cout << test_images.size() << std::endl;
    // std::cout << test_labels.size() << std::endl;

    return 0;
}
