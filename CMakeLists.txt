cmake_minimum_required(VERSION 3.28.3)

project(QALSH-Chamfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------ Dataset Generator ------------

add_executable(dataset_generator 
    src/dataset_generator.cc
    src/dataset_generator_main.cc
)

target_include_directories(dataset_generator PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/third_party"
)

# ------------ Indexer ------------

add_executable(indexer
    src/indexer.cc
    src/indexer_main.cc
)

target_include_directories(indexer PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/third_party"
)


# ------------ QALSH Chamfer ------------

add_executable(qalsh_chamfer
    src/qalsh_chamfer.cc
    src/qalsh_chamfer_main.cc
)

target_include_directories(qalsh_chamfer PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/third_party"
)