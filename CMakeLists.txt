cmake_minimum_required(VERSION 3.28.3)

project(QALSH-Chamfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------ Dataset Generator ------------

add_executable(dataset-generator 
    src/dataset-generator.cc
    src/dataset-generator-main.cc
)

target_include_directories(dataset-generator PRIVATE 
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/third_party"
)