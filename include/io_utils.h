#ifndef IO_UTILS_H
#define IO_UTILS_H

#include <vector>
#include <string>
#include <tuple>
#include <cstdint>

void print_image(std::vector<uint8_t>& image, int& rows, int& cols);

std::tuple<std::vector<std::vector<uint8_t>>, int, int> read_images(const std::string& filename);

std::vector<uint8_t> read_labels(const std::string& filename);

#endif // IO_UTILS_H
