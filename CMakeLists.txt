cmake_minimum_required(VERSION 3.10)
project(ImageRecognition)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

# Automatically find all .cpp files in the src directory
file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(ImageRecognition ${SOURCES})
