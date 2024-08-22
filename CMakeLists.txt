cmake_minimum_required(VERSION 3.10)
project(ImageRecognition LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")

# Use GLOB_RECURSE to find all .cpp and .cu files
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")
message("SOURCES BEFORE: ${SOURCES}")

# Exclude the main file from the test SOURCES list
set(MAIN_FILE "src/main.cpp")
set(TEST_FILE "src/test.cpp")
list(FILTER SOURCES EXCLUDE REGEX ${MAIN_FILE})
list(FILTER SOURCES EXCLUDE REGEX ${TEST_FILE})
message("SOURCES: ${SOURCES}")

# ----------------------------------- MAIN ------------------------------------
# Add the main executable (including main.cpp and test.cpp)
add_executable(ImageRecognition ${SOURCES} ${MAIN_FILE})
# Ensure CUDA code is compiled and linked properly
set_target_properties(ImageRecognition PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)  # Allows for separate compilation of CUDA code
# Link CUDA libraries if needed
find_package(CUDA REQUIRED)
target_link_libraries(ImageRecognition ${CUDA_LIBRARIES})
# Include directories for headers
target_include_directories(ImageRecognition PRIVATE include)

# ----------------------------------- TEST ------------------------------------
# Add the test executable (excluding main.cpp and test.cpp)
add_executable(ImageRecognitionTest ${SOURCES} ${TEST_FILE})
message("SOURCES TEST: ${SOURCES}")
# Ensure CUDA code is compiled and linked properly for the test
set_target_properties(ImageRecognitionTest PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
# Link CUDA libraries for the test executable if needed
target_link_libraries(ImageRecognitionTest ${CUDA_LIBRARIES})
# Include directories for headers in the test executable
target_include_directories(ImageRecognitionTest PRIVATE include)
