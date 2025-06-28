cmake_minimum_required(VERSION 3.28.3)

project(QALSH-Chamfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
    cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.5.0
)

FetchContent_MakeAvailable(cli11)

add_executable(qalsh-chamfer src/main.cpp)

target_link_libraries(qalsh-chamfer PRIVATE CLI11::CLI11)