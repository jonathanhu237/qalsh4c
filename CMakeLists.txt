cmake_minimum_required(VERSION 3.28.3)

project(QALSH-Chamfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -fsanitize=address -fsanitize=undefined")

find_package(CLI11 CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(tomlplusplus CONFIG REQUIRED)

add_executable(qalsh_chamfer 
    src/ann_searcher.cc
    src/b_plus_tree.cc
    src/command.cc
    src/dataset_generator.cc
    src/estimator.cc
    src/global.cc
    src/hash_table_searcher.cc
    src/indexer.cc
    src/main.cc
    src/point_set.cc
    src/quadtree.cc
    src/timer.cc
    src/types.cc
    src/utils.cc
    src/weights_generator.cc
)

target_link_libraries(qalsh_chamfer PRIVATE 
    CLI11::CLI11
    spdlog::spdlog
    Eigen3::Eigen
    tomlplusplus::tomlplusplus
)
