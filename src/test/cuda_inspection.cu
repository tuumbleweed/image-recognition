#include <stdio.h>
#include <iostream>

__global__ void inspect_cuda_3d() {
    printf(
        "threadIdx.x - %d, threadIdx.y - %d, blockIdx.x - %d\n",
        threadIdx.x, threadIdx.y, blockIdx.x
    );
}

__global__ void inspect_cuda_2d() {
    printf(
        "threadIdx.x - %d, blockIdx.x - %d\n",
        threadIdx.x, blockIdx.x
    );
}

// Device function to calculate string length
__device__ int device_strlen(const char *str) {
    int length = 0;
    while (str[length] != '\0') {
        length++;
    }
    return length;
}

__global__ void inspect_update_weights(int current_size, int prev_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    char buffer[100];
    if (idx < current_size) {
        for (int k = 0; k < prev_size; ++k) {
            // accumulate k values
            // Append the single-digit integer as a char to the buffer
            int position = device_strlen(buffer);
            buffer[position] = '0' + k;  // Converts 7 to '7'
            buffer[position + 1] = '\0';     // Null-terminate the string
        }
    }
    printf(
        "blockIdx.x - %d, blockDim.x - %d, threadIdx.x - %d, idx: %d, k: %s\n",
        blockIdx.x, blockDim.x, threadIdx.x, idx, buffer
    );
    // print all
}


int main() {
    double *d_data;
    cudaMalloc(&d_data, sizeof(double));

    // dim3 blockSize(5, 3);
    // int gridSize = 2;
    // inspect_cuda_3d<<<gridSize, blockSize>>>();

    // inspect_cuda_2d<<<3, 5>>>();

    // inspect_update_weights<<<3, 5>>>();

    // inspect_update_weights<<<(2 + 10) / 10, 5>>>();
    // std::cout << (2 + 10) / 10 << "\n"; // 1

    inspect_update_weights<<<3, 5>>>(5, 8);

    cudaFree(d_data);
}
