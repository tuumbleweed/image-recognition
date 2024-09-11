#include "io/normalize.h"

#include "io/colors.h"
#include "io/log.h"

// Function to normalize a vector of uint8_t values to a vector of float values
std::vector<float> normalize(const std::vector<uint8_t>& input) {
    std::vector<float> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = static_cast<float>(input[i]) / 255.0f;
    }
    return output;
}

// Function to normalize all images
std::vector<std::vector<float>> normalize_all(const std::vector<std::vector<uint8_t>>& images) {
    LOG(SL::NOTICE) << "Normalizing " << BLUE << images.size() << RESET << " images" << std::endl;
    std::vector<std::vector<float>> normalized_images;
    normalized_images.reserve(images.size());

    for (const std::vector<uint8_t>& image : images) {
        std::vector<float> normalized_image;
        normalized_image.reserve(image.size());

        for (auto pixel : image) {
            normalized_image.push_back(pixel / 255.0f);
        }

        normalized_images.push_back(std::move(normalized_image));
    }

    return normalized_images;
}
