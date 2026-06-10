cmake_minimum_required(VERSION 3.28.3)

project(QALSH-Chamfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(QALSH4C_ENABLE_CUDA "AUTO" CACHE STRING "Enable CUDA ground-truth acceleration: AUTO, ON, or OFF")
set_property(CACHE QALSH4C_ENABLE_CUDA PROPERTY STRINGS AUTO ON OFF)
string(TOUPPER "${QALSH4C_ENABLE_CUDA}" QALSH4C_ENABLE_CUDA_MODE)
set(QALSH4C_CUDA_MODES AUTO ON OFF)
if(NOT QALSH4C_ENABLE_CUDA_MODE IN_LIST QALSH4C_CUDA_MODES)
    message(FATAL_ERROR "QALSH4C_ENABLE_CUDA must be AUTO, ON, or OFF")
endif()

include(CTest)

include(cmake/Dependencies.cmake)

set(QALSH4C_CUDA_ENABLED OFF)
if(NOT QALSH4C_ENABLE_CUDA_MODE STREQUAL "OFF")
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(QALSH4C_CUDA_ENABLED ON)
    elseif(QALSH4C_ENABLE_CUDA_MODE STREQUAL "ON")
        message(FATAL_ERROR "QALSH4C_ENABLE_CUDA=ON requires a CUDA compiler")
    else()
        message(STATUS "CUDA compiler not found; building CPU-only ground-truth backend")
    endif()
endif()

if(QALSH4C_CUDA_ENABLED)
    set(QALSH4C_DATASET_CUDA_SOURCE src/dataset_cuda.cu)
else()
    set(QALSH4C_DATASET_CUDA_SOURCE src/dataset_cuda_stub.cc)
endif()

function(qalsh4c_configure_cuda_backend target_name)
    if(QALSH4C_CUDA_ENABLED)
        target_compile_definitions(${target_name} PRIVATE QALSH4C_USE_CUDA)
    endif()
endfunction()

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
    ${QALSH4C_DATASET_CUDA_SOURCE}
    src/estimator.cc
    src/global.cc
    src/main.cc
    src/point_set.cc
    src/utils.cc
    src/weights_generator.cc
)
qalsh4c_configure_cuda_backend(qalsh4c)

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
        ${QALSH4C_DATASET_CUDA_SOURCE}
        src/global.cc
        src/point_set.cc
        src/utils.cc
    )
    target_include_directories(dataset_core_test PRIVATE src)
    target_link_libraries(dataset_core_test PRIVATE Eigen3::Eigen nlohmann_json::nlohmann_json)
    qalsh4c_configure_cuda_backend(dataset_core_test)
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
        ${QALSH4C_DATASET_CUDA_SOURCE}
        src/global.cc
        src/point_set.cc
        src/utils.cc
    )
    target_include_directories(dataset_loader_test PRIVATE src)
    target_link_libraries(dataset_loader_test PRIVATE Eigen3::Eigen nlohmann_json::nlohmann_json)
    qalsh4c_configure_cuda_backend(dataset_loader_test)
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
        ${QALSH4C_DATASET_CUDA_SOURCE}
        src/global.cc
        src/point_set.cc
        src/utils.cc
    )
    target_include_directories(dataset_bad_bmp_test PRIVATE src)
    target_link_libraries(dataset_bad_bmp_test PRIVATE Eigen3::Eigen nlohmann_json::nlohmann_json)
    qalsh4c_configure_cuda_backend(dataset_bad_bmp_test)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(dataset_bad_bmp_test PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(dataset_bad_bmp_test PRIVATE QALSH4C_USE_OPENMP)
    endif()
    add_test(NAME dataset_bad_bmp_test COMMAND dataset_bad_bmp_test)
    set_tests_properties(dataset_bad_bmp_test PROPERTIES WILL_FAIL TRUE)

    if(QALSH4C_CUDA_ENABLED)
        add_executable(
            dataset_cuda_backend_test
            tests/dataset_cuda_backend_test.cc
            src/config.cc
            src/dataset.cc
            ${QALSH4C_DATASET_CUDA_SOURCE}
            src/global.cc
            src/point_set.cc
            src/utils.cc
        )
        target_include_directories(dataset_cuda_backend_test PRIVATE src)
        target_link_libraries(dataset_cuda_backend_test PRIVATE Eigen3::Eigen nlohmann_json::nlohmann_json)
        qalsh4c_configure_cuda_backend(dataset_cuda_backend_test)
        add_test(NAME dataset_cuda_backend_test COMMAND dataset_cuda_backend_test)
        set_tests_properties(dataset_cuda_backend_test PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(
            dataset_cuda_cli_test
            tests/dataset_cuda_cli_test.cc
            src/config.cc
            src/global.cc
            src/utils.cc
        )
        target_include_directories(dataset_cuda_cli_test PRIVATE src)
        target_link_libraries(dataset_cuda_cli_test PRIVATE Eigen3::Eigen nlohmann_json::nlohmann_json)
        add_test(
            NAME dataset_cuda_cli_test
            COMMAND dataset_cuda_cli_test $<TARGET_FILE:qalsh4c> ${CMAKE_CURRENT_BINARY_DIR}/dataset_cuda_cli_test_work
        )
        set_tests_properties(dataset_cuda_cli_test PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    add_test(
        NAME qalsh4c_help
        COMMAND ${CMAKE_COMMAND}
            -DQALSH4C_BIN=$<TARGET_FILE:qalsh4c>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_qalsh4c_help.cmake
    )

    add_test(
        NAME qalsh4c_in_memory_qalsh
        COMMAND ${CMAKE_COMMAND}
            -DQALSH4C_BIN=$<TARGET_FILE:qalsh4c>
            -DQALSH4C_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            -DQALSH4C_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_qalsh4c_in_memory_qalsh.cmake
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
        NAME qalsh_ann_help
        COMMAND ${CMAKE_COMMAND}
            -DQALSH4C_BIN=$<TARGET_FILE:qalsh4c>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_qalsh_ann_help.cmake
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
        NAME qalsh4c_dataset_device
        COMMAND ${CMAKE_COMMAND}
            -DQALSH4C_BIN=$<TARGET_FILE:qalsh4c>
            -DQALSH4C_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
            -DQALSH4C_CUDA_ENABLED=${QALSH4C_CUDA_ENABLED}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_qalsh4c_dataset_device.cmake
    )

    add_test(
        NAME qalsh4c_dataset_convert_seed
        COMMAND ${CMAKE_COMMAND}
            -DQALSH4C_BIN=$<TARGET_FILE:qalsh4c>
            -DQALSH4C_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_qalsh4c_dataset_convert_seed.cmake
    )
endif()
