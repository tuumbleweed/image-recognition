#include <stdio.h>

__global__ void test_atomicAdd(double* data) {
    int idx = threadIdx.x;
    if (idx < 5) {
        // data[0] -= 1;
        atomicAdd(&data[0], -1.0f);  // Subtract 1 from data[0]
        printf("-1\n");
    }
}

int main() {
    double *d_data;
    double h_data = 10.0f;
    cudaMalloc(&d_data, sizeof(double));
    cudaMemcpy(d_data, &h_data, sizeof(double), cudaMemcpyHostToDevice);

    test_atomicAdd<<<1, 32>>>(d_data);  // Launch kernel with 32 threads
    cudaMemcpy(&h_data, d_data, sizeof(double), cudaMemcpyDeviceToHost);

    printf("Result: %f\n", h_data);  // Should print 9.0 if atomicAdd worked
    cudaFree(d_data);
    return 0;
}