// cuda_errors.cu
#include <cuda_runtime.h>
#include <execinfo.h>  // For backtrace

#include <cstdlib>
#include <iostream>

#include "io/colors.h"
#include "io/log.h"

inline void printBacktrace() {
    const int maxFrames = 100;
    void* frames[maxFrames];
    int frameCount = backtrace(frames, maxFrames);
    char** symbols = backtrace_symbols(frames, frameCount);

    LOG(SL::ERROR) << GREY << "Backtrace:" << std::endl;
    for (int i = 0; i < frameCount; ++i) {
        LOG(SL::ERROR) << symbols[i] << std::endl;
    }
    LOG(SL::ERROR) << RESET;

    free(symbols);
}

inline void checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        LOG(SL::ERROR) << RED << "\nCUDA Error: " << cudaGetErrorString(error) << " at " << file << ":" << line << RESET
                       << std::endl;
        printBacktrace();
        std::exit(EXIT_FAILURE);
    }
}

// Macro to simplify usage of the inline function
#define CUDA_CHECK_ERROR(call) checkCudaError((call), __FILE__, __LINE__)

// Wrapper for cudaMalloc with the same signature
void cudaMallocWrapper(void** ptr, size_t size) { CUDA_CHECK_ERROR(cudaMalloc(ptr, size)); }

// Wrapper for cudaMemcpy
void cudaMemcpyWrapper(void* dst, const void* src, size_t size, cudaMemcpyKind kind) {
    CUDA_CHECK_ERROR(cudaMemcpy(dst, src, size, kind));
}

// Wrapper for cudaFree
void cudaFreeWrapper(void* ptr) { CUDA_CHECK_ERROR(cudaFree(ptr)); }
