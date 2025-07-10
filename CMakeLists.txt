cmake_minimum_required(VERSION 3.28.3)

project(QALSH-Chamfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -Wpedantic -Wconversion -Wshadow")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fsanitize=undefined")

find_package(CLI11 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(tomlplusplus CONFIG REQUIRED)

add_library(qalsh_chamfer_lib
    src/generate_dataset_command.cc
)

target_link_libraries(qalsh_chamfer_lib PUBLIC
    CLI11::CLI11
)

target_include_directories(qalsh_chamfer_lib PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
)

add_executable(qalsh_chamfer src/main.cc)
target_link_libraries(qalsh_chamfer PRIVATE qalsh_chamfer_lib)
