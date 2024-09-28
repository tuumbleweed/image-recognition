cmake_minimum_required(VERSION 3.10)
project(ImageRecognition LANGUAGES CXX CUDA)

# Add FMA flag for GCC or Clang
add_compile_options(-mfma)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")

# Use GLOB_RECURSE to find all .cpp and .cu files, including module files src/.../*.cpp
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")
message("SOURCES BEFORE: ${SOURCES}")

# Exclude the main file from the matrix mult SOURCES list
set(MAIN_FILE "src/main.cpp")
set(MATRIX_MULT_FILE "src/test/matrix_mult.cpp")
set(ATOMIC_FILE "src/test/atomic.cu")
set(CUDA_INSPECTION_FILE "src/test/cuda_inspection.cu")
list(FILTER SOURCES EXCLUDE REGEX ${MAIN_FILE})
list(FILTER SOURCES EXCLUDE REGEX ${MATRIX_MULT_FILE})
list(FILTER SOURCES EXCLUDE REGEX ${ATOMIC_FILE})
list(FILTER SOURCES EXCLUDE REGEX ${CUDA_INSPECTION_FILE})
message("SOURCES: ${SOURCES}")

# ----------------------------------- DEPENDENCIES ------------------------------------
# Find Boost
find_package(Boost 1.65 REQUIRED COMPONENTS program_options)
include_directories(${Boost_INCLUDE_DIRS})

# ----------------------------------- MAIN ------------------------------------
# Add the main executable (including main.cpp and matrix mult.cpp)
add_executable(ImageRecognition ${SOURCES} ${MAIN_FILE})
# Ensure CUDA code is compiled and linked properly
set_target_properties(ImageRecognition PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)  # Allows for separate compilation of CUDA code
# Link CUDA libraries if needed
find_package(CUDA REQUIRED)
target_link_libraries(ImageRecognition ${CUDA_LIBRARIES} ${Boost_LIBRARIES})
# Include directories for headers
target_include_directories(ImageRecognition PRIVATE include)

# ----------------------------------- MATRIX MULT ------------------------------------
# Add the matrix mult executable (excluding main.cpp and matrix_mult.cpp)
add_executable(MatrixMult ${SOURCES} ${MATRIX_MULT_FILE})
add_executable(Atomic ${ATOMIC_FILE})
add_executable(CudaInspection ${CUDA_INSPECTION_FILE})
message("SOURCES matrix mult: ${SOURCES}")
# Ensure CUDA code is compiled and linked properly for the matrix mult
set_target_properties(MatrixMult PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
# Link CUDA libraries for the matrix mult executable if needed
target_link_libraries(MatrixMult ${CUDA_LIBRARIES} ${Boost_LIBRARIES})
# Include directories for headers in the matrix mult executable
target_include_directories(MatrixMult PRIVATE include)
