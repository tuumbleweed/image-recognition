#include <cuda_runtime.h>

#include <iostream>
#include <cstdlib> // For the exit function

#include "gpu.h"
#include "log.h"


// Function to perform matrix multiplication on the GPU
std::vector<float> multiply_on_gpu(const std::vector<float>& A, const std::vector<std::vector<float>>& B, const std::vector<float>& biases) {
    // Flatten matrix B into a single array
    std::vector<float> B_flat;
    for (const auto& row : B) {
        B_flat.insert(B_flat.end(), row.begin(), row.end());
    }

    int a_length = A.size();
    int b_flat_length = B_flat.size();
    int b_length = B.size();    // Number of rows in B
    int b_width = B[0].size();  // Number of columns in B
    int biases_length = biases.size();
    // std::cout << "a_length: " << a_length << std::endl;
    // std::cout << "b_flat_length: " << b_flat_length << std::endl;
    // std::cout << "b_length: " << b_length << std::endl;
    // std::cout << "b_width: " << b_width << std::endl;
    if (a_length != b_width) {
        std::cerr << "a_length != b_width, " << a_length << " != " << b_width << std::endl;
        exit(EXIT_FAILURE);
    }
    int a_length_b_width = b_width;

    // Allocate memory on the device (GPU)
    float *d_A, *d_B, *d_biases, *d_result;
    cudaMalloc(&d_A, a_length * sizeof(float));
    cudaMalloc(&d_B, b_flat_length * sizeof(float));
    cudaMalloc(&d_biases, biases_length * sizeof(float));
    cudaMalloc(&d_result, b_length * sizeof(float));

    // Copy data from host (CPU) to device (GPU)
    cudaMemcpy(d_A, A.data(), a_length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_flat.data(), b_flat_length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, biases.data(), biases_length * sizeof(float), cudaMemcpyHostToDevice);

    // Define the grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((b_width + threadsPerBlock.x - 1) / threadsPerBlock.x, 1);

    // Launch the kernel
    matrix_multiply_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_biases, d_result, a_length_b_width, b_length);

    // Copy the result back to the host
    std::vector<float> output(b_length);
    cudaMemcpy(output.data(), d_result, b_length * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_biases);
    cudaFree(d_result);

    return output;
}
