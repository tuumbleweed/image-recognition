#include <stdio.h>

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

int main() {
    float *d_data;
    cudaMalloc(&d_data, sizeof(float));

    // dim3 blockSize(5, 3);
    // int gridSize = 2;
    // inspect_cuda_3d<<<gridSize, blockSize>>>();

    inspect_cuda_2d<<<3, 5>>>();


    cudaFree(d_data);
}
