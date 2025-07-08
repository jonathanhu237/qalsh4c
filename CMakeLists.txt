cmake_minimum_required(VERSION 3.28.3)

project(QALSH-Chamfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -fsanitize=address -fsanitize=undefined")

add_library(qalsh_chamfer_lib
    src/b_plus_tree.cc
    src/chamfer_estimator.cc
    src/dataset_generator.cc
    src/indexer.cc
    src/utils.cc
)

target_include_directories(qalsh_chamfer_lib PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/third_party"
)

add_executable(chamfer_estimator src/chamfer_estimator_main.cc)
add_executable(dataset_generator src/dataset_generator_main.cc)
add_executable(indexer src/indexer_main.cc)

target_link_libraries(chamfer_estimator PRIVATE qalsh_chamfer_lib)
target_link_libraries(dataset_generator PRIVATE qalsh_chamfer_lib)
target_link_libraries(indexer PRIVATE qalsh_chamfer_lib)