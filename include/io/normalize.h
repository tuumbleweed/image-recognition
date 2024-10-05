#ifndef NORMALIZE_H
#define NORMALIZE_H

#include <vector>
#include <cstdint>

// Function to normalize a vector of uint8_t values to a vector of double values
std::vector<double> normalize(const std::vector<uint8_t>& input);

// Function to normalize all images
std::vector<std::vector<double>> normalize_all(const std::vector<std::vector<uint8_t>>& images);

#endif // NORMALIZE_H
