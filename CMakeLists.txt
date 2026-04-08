cmake_minimum_required(VERSION 3.28.3)

project(QALSH-Chamfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)

find_package(CLI11 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

add_executable(qalsh4c 
    src/ann_searcher.cc
    src/b_plus_tree.cc
    src/buffer.cc
    src/command.cc
    src/config.cc
    src/estimator.cc
    src/global.cc
    src/main.cc
    src/point_set.cc
    src/utils.cc
    src/weights_generator.cc
)

target_link_libraries(qalsh4c PRIVATE 
    CLI11::CLI11
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
)

if(BUILD_TESTING)
    add_executable(log_smoke_test tests/log_smoke_test.cc)
    target_include_directories(log_smoke_test PRIVATE src)
    add_test(NAME log_smoke_test COMMAND log_smoke_test)

    add_executable(log_error_test tests/log_error_test.cc)
    target_include_directories(log_error_test PRIVATE src)
    add_test(NAME log_error_test COMMAND log_error_test)
    set_tests_properties(log_error_test PROPERTIES WILL_FAIL TRUE)

    add_test(
        NAME qalsh4c_help
        COMMAND ${CMAKE_COMMAND}
            -DQALSH4C_BIN=$<TARGET_FILE:qalsh4c>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_qalsh4c_help.cmake
    )
endif()
