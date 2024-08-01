# Specify the project and enable CUDA
cmake_minimum_required(VERSION 3.10)
project(ImageRecognition LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use GLOB_RECURSE to find all .cpp and .cu files
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")

# Add the executable
add_executable(ImageRecognition ${SOURCES})

# Ensure CUDA code is compiled and linked properly
set_target_properties(ImageRecognition PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)  # Allows for separate compilation of CUDA code

# Link CUDA libraries if needed
find_package(CUDA REQUIRED)
target_link_libraries(ImageRecognition ${CUDA_LIBRARIES})

# Include directories for headers
target_include_directories(ImageRecognition PRIVATE include)
