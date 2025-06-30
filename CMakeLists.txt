cmake_minimum_required(VERSION 3.28.3)

project(QALSH-Chamfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(qalsh_chamfer_lib
    src/dataset_generator.cc
    src/indexer.cc
    src/qalsh_chamfer.cc
)

target_include_directories(qalsh_chamfer_lib PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/third_party"
)

add_executable(dataset_generator src/dataset_generator_main.cc)
add_executable(indexer src/indexer_main.cc)
add_executable(qalsh_chamfer src/qalsh_chamfer_main.cc)

target_link_libraries(dataset_generator PRIVATE qalsh_chamfer_lib)
target_link_libraries(indexer PRIVATE qalsh_chamfer_lib)
target_link_libraries(qalsh_chamfer PRIVATE qalsh_chamfer_lib)
