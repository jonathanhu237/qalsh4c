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

add_subdirectory(src/qalsh)

add_executable(qalsh4c 
    src/ann_searcher.cc
    src/exact_scan.cc
    src/command.cc
    src/config.cc
    src/estimator.cc
    src/global.cc
    src/kappa.cc
    src/main.cc
    src/point_set.cc
    src/utils.cc
    src/weights_generator.cc
)
# The kappa certificate widens every floating-point result by one ulp; fused multiply-adds would bypass that.
set_source_files_properties(src/kappa.cc PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")

target_link_libraries(qalsh4c PRIVATE 
    CLI11::CLI11
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    qalsh4c_qalsh
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(qalsh4c PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(qalsh4c PRIVATE QALSH4C_USE_OPENMP)
endif()

if(BUILD_TESTING)
    add_executable(kappa_test tests/kappa_test.cc src/kappa.cc)
    target_include_directories(kappa_test PRIVATE src)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(kappa_test PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(kappa_test PRIVATE QALSH4C_USE_OPENMP)
    endif()
    add_test(NAME kappa_test COMMAND kappa_test)

    add_executable(distance_bounds_test tests/distance_bounds_test.cc src/utils.cc src/global.cc)
    target_include_directories(distance_bounds_test PRIVATE src)
    target_link_libraries(distance_bounds_test PRIVATE Eigen3::Eigen nlohmann_json::nlohmann_json)
    add_test(NAME distance_bounds_test COMMAND distance_bounds_test)

    add_executable(log_smoke_test tests/log_smoke_test.cc)
    target_include_directories(log_smoke_test PRIVATE src)
    add_test(NAME log_smoke_test COMMAND log_smoke_test)

    add_executable(
        ann_searcher_clone_test
        tests/ann_searcher_clone_test.cc
        src/ann_searcher.cc
        src/exact_scan.cc
        src/config.cc
            src/global.cc
        src/point_set.cc
        src/utils.cc
    )
    target_include_directories(ann_searcher_clone_test PRIVATE src)
    target_link_libraries(ann_searcher_clone_test PRIVATE Eigen3::Eigen nlohmann_json::nlohmann_json qalsh4c_qalsh)
    add_test(NAME ann_searcher_clone_test COMMAND ann_searcher_clone_test)

    add_executable(
        exact_scan_test
        tests/exact_scan_test.cc
        src/ann_searcher.cc
        src/exact_scan.cc
        src/config.cc
            src/global.cc
        src/point_set.cc
        src/utils.cc
    )
    target_include_directories(exact_scan_test PRIVATE src)
    target_link_libraries(exact_scan_test PRIVATE Eigen3::Eigen nlohmann_json::nlohmann_json qalsh4c_qalsh)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(exact_scan_test PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(exact_scan_test PRIVATE QALSH4C_USE_OPENMP)
    endif()
    add_test(NAME exact_scan_test COMMAND exact_scan_test)

    add_executable(
        quadtree_weights_generator_test
        tests/quadtree_weights_generator_test.cc
        src/ann_searcher.cc
        src/exact_scan.cc
        src/config.cc
            src/global.cc
        src/point_set.cc
        src/utils.cc
        src/weights_generator.cc
    )
    target_include_directories(quadtree_weights_generator_test PRIVATE src)
    target_link_libraries(quadtree_weights_generator_test PRIVATE Eigen3::Eigen nlohmann_json::nlohmann_json qalsh4c_qalsh)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(quadtree_weights_generator_test PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(quadtree_weights_generator_test PRIVATE QALSH4C_USE_OPENMP)
    endif()
    add_test(NAME quadtree_weights_generator_test COMMAND quadtree_weights_generator_test)

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
        NAME qalsh4c_qalsh_weights_cache
        COMMAND ${CMAKE_COMMAND}
            -DQALSH4C_BIN=$<TARGET_FILE:qalsh4c>
            -DQALSH4C_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            -DQALSH4C_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_qalsh4c_qalsh_weights_cache.cmake
    )

    add_test(
        NAME qalsh4c_quadtree_sampling
        COMMAND ${CMAKE_COMMAND}
            -DQALSH4C_BIN=$<TARGET_FILE:qalsh4c>
            -DQALSH4C_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            -DQALSH4C_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_qalsh4c_quadtree_sampling.cmake
    )

    add_test(
        NAME qalsh4c_lookup
        COMMAND ${CMAKE_COMMAND}
            -DQALSH4C_BIN=$<TARGET_FILE:qalsh4c>
            -DQALSH4C_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            -DQALSH4C_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_qalsh4c_lookup.cmake
    )
endif()
