#include "gpu.h"
#include <cuda_runtime.h>

// Function to perform matrix multiplication on the GPU
std::vector<float> multiply_on_gpu(const std::vector<float>& A, const std::vector<std::vector<float>>& B) {
	// Flatten matrix B into a single array
	std::vector<float> B_flat;
	for (const auto& row : B) {
		B_flat.insert(B_flat.end(), row.begin(), row.end());
	}

	int A_length = A.size();
	int B_length = B_flat.size();
	int b_length = B.size(); // Number of rows in B
	int b_width = B[0].size(); // Number of columns in B

	// Allocate memory on the device (GPU)
	float *d_A, *d_B, *d_result;
	cudaMalloc(&d_A, A_length * sizeof(float));
	cudaMalloc(&d_B, B_length * sizeof(float));
	cudaMalloc(&d_result, b_width * sizeof(float));

	// Copy data from host (CPU) to device (GPU)
	cudaMemcpy(d_A, A.data(), A_length * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B_flat.data(), B_length * sizeof(float), cudaMemcpyHostToDevice);

	// Define the grid and block dimensions
	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((b_width + threadsPerBlock.x - 1) / threadsPerBlock.x, 1);

	// Launch the kernel
	matrix_multiply_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_result, b_length, b_width);

	// Copy the result back to the host
	std::vector<float> output(b_width);
	cudaMemcpy(output.data(), d_result, b_width * sizeof(float), cudaMemcpyDeviceToHost);

	// Free GPU memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_result);

	return output;
}
