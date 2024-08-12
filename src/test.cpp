#include "cpu.h"
#include "gpu.h"
#include <iostream>

std::vector<float> multiply_on_gpu(const std::vector<float>& A, const std::vector<std::vector<float>>& B);

// Function for matrix multiplication
// matrix1 is one dimentional, matrix2 is 2D
// matrix1 length should be equal to matrix2 length
// (number of float elements in matrix1 == number of std::vector<float> in matrix2)
std::vector<float> matrix_multiply_cpu_control(const std::vector<float>& matrix1, const std::vector<std::vector<float>>& matrix2) {
    int matrix1_length = matrix1.size();
    int matrix2_width = matrix2[0].size();
    int matrix2_length = matrix2.size();
    std::vector<float> output_matrix(matrix2_width);

    std::cout << "matrix_multiply_cpu_proper operations:\n";
    for (int matrix2_col = 0; matrix2_col < matrix2_width; matrix2_col++) {
        float sum = 0.0f;
        for (int matrix1_index=0; matrix1_index < matrix1_length; matrix1_index++) {
            float value = matrix1[matrix1_index] * matrix2[matrix1_index][matrix2_col];
            sum += value;
            std::cout << matrix1[matrix1_index] << "*" << matrix2[matrix1_index][matrix2_col] << "=" << value << " ";
        }
        output_matrix[matrix2_col] = sum;
        std::cout << "| == " << sum << std::endl;
    }

    return output_matrix;
}

int main() {
    // tet what chentd
    std::vector<float> A = {0.5, 1.2, -0.7, 3.4};
    std::vector<std::vector<float>> B = {
        {0.1, -0.4, 2.3},
        {1.5, 0.0, -1.2},
        {-0.3, 4.1, 2.2},
        {0.7, -0.8, 1.9}
    };

    std::vector<float> cpu_control_output = matrix_multiply_cpu_control(A, B);
    std::vector<float> cpu_output = matrix_multiply_cpu(A, B, 4, 3);
	std::vector<float> gpu_output = multiply_on_gpu(A, B);
    std::cout << "matrix_multiply_cpu_control: ";
    for (float& value : cpu_control_output) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
    std::cout << "matrix_multiply_cpu: ";
    for (float& value : cpu_output) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

	// Print the result
	std::cout << "multiply_on_gpu: ";
	for (float& value : gpu_output) {
		std::cout << value << " ";
	}
	std::cout << std::endl;

    return 0;
}
