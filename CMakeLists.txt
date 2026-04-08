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
    src/external_sorter.cc
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

    add_executable(external_sorter_test tests/external_sorter_test.cc src/external_sorter.cc)
    target_include_directories(external_sorter_test PRIVATE src)
    add_test(NAME external_sorter_test COMMAND external_sorter_test)

    add_executable(
        b_plus_tree_stream_test
        tests/b_plus_tree_stream_test.cc
        src/b_plus_tree.cc
        src/buffer.cc
    )
    target_include_directories(b_plus_tree_stream_test PRIVATE src)
    add_test(NAME b_plus_tree_stream_test COMMAND b_plus_tree_stream_test)

    add_executable(
        b_plus_tree_searcher_cache_test
        tests/b_plus_tree_searcher_cache_test.cc
        src/b_plus_tree.cc
        src/buffer.cc
    )
    target_include_directories(b_plus_tree_searcher_cache_test PRIVATE src)
    add_test(NAME b_plus_tree_searcher_cache_test COMMAND b_plus_tree_searcher_cache_test)

    add_executable(
        qalsh_ann_candidate_limit_test
        tests/qalsh_ann_candidate_limit_test.cc
        src/ann_searcher.cc
        src/b_plus_tree.cc
        src/buffer.cc
        src/config.cc
        src/global.cc
        src/point_set.cc
        src/utils.cc
    )
    target_include_directories(qalsh_ann_candidate_limit_test PRIVATE src)
    target_link_libraries(qalsh_ann_candidate_limit_test PRIVATE Eigen3::Eigen nlohmann_json::nlohmann_json)
    add_test(NAME qalsh_ann_candidate_limit_test COMMAND qalsh_ann_candidate_limit_test)

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

    add_test(
        NAME qalsh4c_index_modes
        COMMAND ${CMAKE_COMMAND}
            -DQALSH4C_BIN=$<TARGET_FILE:qalsh4c>
            -DQALSH4C_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            -DQALSH4C_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_qalsh4c_index_modes.cmake
    )

    add_test(
        NAME qalsh_ann_help
        COMMAND ${CMAKE_COMMAND}
            -DQALSH4C_BIN=$<TARGET_FILE:qalsh4c>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_qalsh_ann_help.cmake
    )
endif()
