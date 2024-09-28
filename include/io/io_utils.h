#ifndef IO_UTILS_H
#define IO_UTILS_H

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "network/init.h"

void print_image(const std::vector<double>& image, const int& rows, const int& cols);

std::tuple<std::vector<std::vector<uint8_t>>, int, int> read_images(const std::string& filename);

std::vector<uint8_t> read_labels(const std::string& filename);

void print_network(const std::vector<Layer>& network);

#endif  // IO_UTILS_H
