#ifndef TRAIN_H
#define TRAIN_H

#include <vector>

#include "layer.h"

void train_on_one_image_cpu(const int counter, const uint8_t& label, const std::vector<float>& train_image, std::vector<Layer>& network);

#endif // TRAIN_H