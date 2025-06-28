cmake_minimum_required(VERSION 3.28.3)

project(QALSH-Chamfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(qalsh-chamfer src/main.cpp)

target_include_directories(qalsh-chamfer PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/vendor")