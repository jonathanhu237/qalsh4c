cmake_minimum_required(VERSION 3.28.3)

project(QALSH-Chamfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CLI11 CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

add_executable(qalsh_chamfer 
    src/ann_searcher.cc
    src/b_plus_tree.cc
    src/command.cc
    src/estimator.cc
    src/global.cc
    src/main.cc
    src/utils.cc
    src/weights_generator.cc
)

target_link_libraries(qalsh_chamfer PRIVATE 
    CLI11::CLI11
    spdlog::spdlog
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
)
