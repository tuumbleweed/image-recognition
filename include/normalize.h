#ifndef NORMALIZE_H
#define NORMALIZE_H

#include <vector>
#include <cstdint>

// Function to normalize a vector of uint8_t values to a vector of float values
std::vector<float> normalize(const std::vector<uint8_t>& input);

// Function to normalize all images
std::vector<std::vector<float>> normalize_all(std::vector<std::vector<uint8_t>>& images);

#endif // NORMALIZE_H
