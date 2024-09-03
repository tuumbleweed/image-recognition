#include <cstdlib>  // For the exit function
#include <iostream>

#include "io/log.h"
#include "network/cpu.h"
#include "network/gpu.h"

std::vector<float> multiply_on_gpu(const std::vector<float>& A, const std::vector<std::vector<float>>& B,
                                   const std::vector<float>& biases);

// Function for matrix multiplication
// matrix1 is one dimentional, matrix2 is 2D
// matrix1 length should be equal to matrix2 length
// (number of float elements in matrix1 == number of std::vector<float> in matrix2)
std::vector<float> matrix_multiply_cpu_control(const std::vector<float>& A, const std::vector<std::vector<float>>& B,
                                               const std::vector<float>& biases) {
    int a_length = A.size();
    int b_length = B.size();
    int b_width = B[0].size();
    LOG(SL::TRACE) << "a_length: " << a_length << std::endl;
    LOG(SL::TRACE) << "b_length: " << b_length << std::endl;
    LOG(SL::TRACE) << "b_width: " << b_width << std::endl;
    if (a_length != b_width) {
        LOG(SL::ERROR) << "a_length != b_width, " << a_length << " != " << b_width << std::endl;
        exit(EXIT_FAILURE);
    }
    int a_length_b_width = b_width;
    std::vector<float> result(b_length);

    LOG(SL::DEBUG) << "matrix_multiply_cpu_proper operations:\n";
    for (int b_row = 0; b_row < b_length; b_row++) {
        float sum = 0.0f;
        float bias_value = biases[b_row];
        for (int ab_index = 0; ab_index < a_length_b_width; ab_index++) {
            float a_value = A[ab_index];
            float b_value = B[b_row][ab_index];
            float value = a_value * b_value;
            sum += value;
            LOG(SL::DEBUG) << a_value << "*" << b_value << "=" << value << " ";
            if (ab_index < a_length_b_width - 1) {
                LOG(SL::DEBUG) << "+ ";
            }
        }
        result[b_row] = sum + bias_value;
        LOG(SL::DEBUG) << "+ " << bias_value << " | == " << sum << std::endl;
    }

    return result;
}

int main() {
    SL::load_config("cfg/config.ini");

    std::vector<float> delta(10, 1.0f);  // Vector of size 10

    // Resize vector to size 16
    delta.resize(8);

    delta[5] = 0.0f;
    LOG(SL::DEBUG) << "Delta: " << delta[5] << " " << delta[7] << std::endl;

    // tet what chentd
    std::vector<float> A = {0.5, 1.2, -0.7, 3.4};
    std::vector<std::vector<float>> B = {{0.1, -0.4, 2.3, 1.5}, {0.0, -1.2, -0.3, 4.1}, {2.2, 0.7, -0.8, 1.9}};
    std::vector<float> biases = {2.0, 1.8, 1.9, 0.1};

    std::vector<float> cpu_control_output = matrix_multiply_cpu_control(A, B, biases);
    std::vector<float> cpu_output = matrix_multiply_cpu(A, B, biases, 4, 3);
    std::vector<float> gpu_output = multiply_on_gpu(A, B, biases);
    LOG(SL::DEBUG) << "matrix_multiply_cpu_control: ";
    for (float& value : cpu_control_output) {
        LOG(SL::DEBUG) << value << " ";
    }
    LOG(SL::DEBUG) << std::endl;
    LOG(SL::DEBUG) << "matrix_multiply_cpu: ";
    for (float& value : cpu_output) {
        LOG(SL::DEBUG) << value << " ";
    }
    LOG(SL::DEBUG) << std::endl;

    // Print the result
    LOG(SL::DEBUG) << "multiply_on_gpu: ";
    for (float& value : gpu_output) {
        LOG(SL::DEBUG) << value << " ";
    }
    LOG(SL::DEBUG) << std::endl;

    return 0;
}
