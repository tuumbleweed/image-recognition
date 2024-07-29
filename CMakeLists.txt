cmake_minimum_required(VERSION 3.10)
project(ImageLabelReader)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

add_executable(ImageLabelReader src/main.cpp src/io_utils.cpp)
