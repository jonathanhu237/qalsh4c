cmake_minimum_required(VERSION 3.28.3)

project(QALSH-Chamfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)

include(cmake/Dependencies.cmake)

if(APPLE AND EXISTS "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    set(OpenMP_CXX_INCLUDE_DIR "/opt/homebrew/opt/libomp/include")
    set(OpenMP_libomp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
endif()

find_package(OpenMP COMPONENTS CXX)

add_executable(qalsh4c 
    src/ann_searcher.cc
    src/command.cc
    src/config.cc
    src/dataset.cc
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

if(OpenMP_CXX_FOUND)
    target_link_libraries(qalsh4c PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(qalsh4c PRIVATE QALSH4C_USE_OPENMP)
endif()

if(BUILD_TESTING)
    add_executable(log_smoke_test tests/log_smoke_test.cc)
    target_include_directories(log_smoke_test PRIVATE src)
    add_test(NAME log_smoke_test COMMAND log_smoke_test)

    add_executable(external_sorter_test tests/external_sorter_test.cc src/external_sorter.cc)
    target_include_directories(external_sorter_test PRIVATE src)
    add_test(NAME external_sorter_test COMMAND external_sorter_test)

    add_executable(external_sorter_prefix_test tests/external_sorter_prefix_test.cc src/external_sorter.cc)
    target_include_directories(external_sorter_prefix_test PRIVATE src)
    add_test(NAME external_sorter_prefix_test COMMAND external_sorter_prefix_test)

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
        src/config.cc
        src/global.cc
        src/point_set.cc
        src/utils.cc
    )
    target_include_directories(qalsh_ann_candidate_limit_test PRIVATE src)
    target_link_libraries(qalsh_ann_candidate_limit_test PRIVATE Eigen3::Eigen nlohmann_json::nlohmann_json)
    add_test(NAME qalsh_ann_candidate_limit_test COMMAND qalsh_ann_candidate_limit_test)

    add_executable(
        ann_searcher_clone_test
        tests/ann_searcher_clone_test.cc
        src/ann_searcher.cc
        src/config.cc
        src/global.cc
        src/point_set.cc
        src/utils.cc
    )
    target_include_directories(ann_searcher_clone_test PRIVATE src)
    target_link_libraries(ann_searcher_clone_test PRIVATE Eigen3::Eigen nlohmann_json::nlohmann_json)
    add_test(NAME ann_searcher_clone_test COMMAND ann_searcher_clone_test)

    add_executable(log_error_test tests/log_error_test.cc)
    target_include_directories(log_error_test PRIVATE src)
    add_test(NAME log_error_test COMMAND log_error_test)
    set_tests_properties(log_error_test PROPERTIES WILL_FAIL TRUE)

    add_executable(
        dataset_core_test
        tests/dataset_core_test.cc
        src/config.cc
        src/dataset.cc
        src/global.cc
        src/point_set.cc
        src/utils.cc
    )
    target_include_directories(dataset_core_test PRIVATE src)
    target_link_libraries(dataset_core_test PRIVATE Eigen3::Eigen nlohmann_json::nlohmann_json)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(dataset_core_test PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(dataset_core_test PRIVATE QALSH4C_USE_OPENMP)
    endif()
    add_test(NAME dataset_core_test COMMAND dataset_core_test)

    add_executable(
        dataset_loader_test
        tests/dataset_loader_test.cc
        src/config.cc
        src/dataset.cc
        src/global.cc
        src/point_set.cc
        src/utils.cc
    )
    target_include_directories(dataset_loader_test PRIVATE src)
    target_link_libraries(dataset_loader_test PRIVATE Eigen3::Eigen nlohmann_json::nlohmann_json)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(dataset_loader_test PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(dataset_loader_test PRIVATE QALSH4C_USE_OPENMP)
    endif()
    add_test(NAME dataset_loader_test COMMAND dataset_loader_test)

    add_executable(
        dataset_bad_bmp_test
        tests/dataset_bad_bmp_test.cc
        src/config.cc
        src/dataset.cc
        src/global.cc
        src/point_set.cc
        src/utils.cc
    )
    target_include_directories(dataset_bad_bmp_test PRIVATE src)
    target_link_libraries(dataset_bad_bmp_test PRIVATE Eigen3::Eigen nlohmann_json::nlohmann_json)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(dataset_bad_bmp_test PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(dataset_bad_bmp_test PRIVATE QALSH4C_USE_OPENMP)
    endif()
    add_test(NAME dataset_bad_bmp_test COMMAND dataset_bad_bmp_test)
    set_tests_properties(dataset_bad_bmp_test PROPERTIES WILL_FAIL TRUE)

    add_test(
        NAME qalsh4c_help
        COMMAND ${CMAKE_COMMAND}
            -DQALSH4C_BIN=$<TARGET_FILE:qalsh4c>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_qalsh4c_help.cmake
    )

    add_test(
        NAME qalsh4c_direction_help
        COMMAND ${CMAKE_COMMAND}
            -DQALSH4C_BIN=$<TARGET_FILE:qalsh4c>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_qalsh4c_direction_help.cmake
    )

    add_test(
        NAME qalsh4c_direction_estimate
        COMMAND ${CMAKE_COMMAND}
            -DQALSH4C_BIN=$<TARGET_FILE:qalsh4c>
            -DQALSH4C_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            -DQALSH4C_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_qalsh4c_direction_estimate.cmake
    )

    add_test(
        NAME qalsh_ann_help
        COMMAND ${CMAKE_COMMAND}
            -DQALSH4C_BIN=$<TARGET_FILE:qalsh4c>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_qalsh_ann_help.cmake
    )

    add_test(
        NAME qalsh4c_qalsh_sampling_weights
        COMMAND ${CMAKE_COMMAND}
            -DQALSH4C_BIN=$<TARGET_FILE:qalsh4c>
            -DQALSH4C_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            -DQALSH4C_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_qalsh4c_qalsh_sampling_weights.cmake
    )

    add_test(
        NAME qalsh4c_dataset_generate
        COMMAND ${CMAKE_COMMAND}
            -DQALSH4C_BIN=$<TARGET_FILE:qalsh4c>
            -DQALSH4C_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_qalsh4c_dataset_generate.cmake
    )

    add_test(
        NAME qalsh4c_dataset_ground_truth
        COMMAND ${CMAKE_COMMAND}
            -DQALSH4C_BIN=$<TARGET_FILE:qalsh4c>
            -DQALSH4C_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_qalsh4c_dataset_ground_truth.cmake
    )

    add_test(
        NAME qalsh4c_dataset_overwrite
        COMMAND ${CMAKE_COMMAND}
            -DQALSH4C_BIN=$<TARGET_FILE:qalsh4c>
            -DQALSH4C_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_qalsh4c_dataset_overwrite.cmake
    )

    add_test(
        NAME qalsh4c_dataset_invalid
        COMMAND ${CMAKE_COMMAND}
            -DQALSH4C_BIN=$<TARGET_FILE:qalsh4c>
            -DQALSH4C_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
            -DQALSH4C_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_qalsh4c_dataset_invalid.cmake
    )

    add_test(
        NAME qalsh4c_dataset_convert_seed
        COMMAND ${CMAKE_COMMAND}
            -DQALSH4C_BIN=$<TARGET_FILE:qalsh4c>
            -DQALSH4C_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_qalsh4c_dataset_convert_seed.cmake
    )
endif()
