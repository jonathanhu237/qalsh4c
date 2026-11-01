option(QALSH_WINDOW_FORCE_SCALAR
       "Force the private window-clamp implementation to use its scalar fallback" OFF)
option(QALSH_WIDE_COLLISION_COUNTERS
       "Keep four-byte collision counters for every query, for the counter-width ablation" OFF)

add_library(qalsh4c_qalsh STATIC src/qalsh.cc)

target_compile_features(qalsh4c_qalsh PUBLIC cxx_std_20)
find_package(Threads REQUIRED)
target_link_libraries(qalsh4c_qalsh PUBLIC Threads::Threads)
target_include_directories(qalsh4c_qalsh
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(qalsh4c_qalsh PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>
)

# Use SIMD reductions when the host toolchain provides the OpenMP SIMD
# front-end, without making the library depend on the OpenMP runtime.
find_package(OpenMP QUIET COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
    target_compile_definitions(qalsh4c_qalsh PRIVATE QALSH_USE_OPENMP_SIMD)
    target_compile_options(qalsh4c_qalsh PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fopenmp-simd>
    )
endif()

# Ablation-build switch: results are identical, only the per-query counter footprint changes.
if(QALSH_WIDE_COLLISION_COUNTERS)
    target_compile_definitions(qalsh4c_qalsh PRIVATE QALSH_WIDE_COLLISION_COUNTERS)
endif()

# This is a private validation-build switch.  It is intentionally off for
# normal consumers and only changes the implementation's internal fallback.
if(QALSH_WINDOW_FORCE_SCALAR)
    target_compile_definitions(qalsh4c_qalsh PRIVATE QALSH_WINDOW_FORCE_SCALAR)
endif()

set_target_properties(qalsh4c_qalsh PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_STANDARD 20)

if(BUILD_TESTING)
    add_executable(qalsh_public_test tests/public_test.cc)
    target_link_libraries(qalsh_public_test PRIVATE qalsh4c_qalsh)
    add_test(NAME qalsh_public_test COMMAND qalsh_public_test)
    add_executable(qalsh_scale_equivalence_test tests/scale_equivalence_test.cc)
    target_link_libraries(qalsh_scale_equivalence_test PRIVATE qalsh4c_qalsh)
    add_test(NAME qalsh_scale_equivalence_test COMMAND qalsh_scale_equivalence_test)
    add_executable(qalsh_dispatch_test tests/dispatch_test.cc)
    target_link_libraries(qalsh_dispatch_test PRIVATE qalsh4c_qalsh)
    add_test(NAME qalsh_dispatch_test COMMAND qalsh_dispatch_test ${CMAKE_CURRENT_BINARY_DIR})
    add_executable(qalsh_distance_test tests/distance_test.cc)
    target_link_libraries(qalsh_distance_test PRIVATE qalsh4c_qalsh)
    add_test(NAME qalsh_distance_test COMMAND qalsh_distance_test)
    add_executable(qalsh_build_test tests/build_test.cc)
    target_link_libraries(qalsh_build_test PRIVATE qalsh4c_qalsh)
    add_test(NAME qalsh_build_test COMMAND qalsh_build_test ${CMAKE_CURRENT_BINARY_DIR})

    add_executable(qalsh_query_speed_test tests/query_speed_test.cc)
    target_link_libraries(qalsh_query_speed_test PRIVATE qalsh4c_qalsh)
    add_test(NAME qalsh_query_speed_test
        COMMAND qalsh_query_speed_test ${CMAKE_CURRENT_BINARY_DIR})

    add_executable(qalsh_collision_counter_test tests/collision_counter_test.cc)
    target_link_libraries(qalsh_collision_counter_test PRIVATE qalsh4c_qalsh)
    add_test(NAME qalsh_collision_counter_test COMMAND qalsh_collision_counter_test)

    add_executable(qalsh_window_clamp_test tests/window_clamp_test.cc)
    target_link_libraries(qalsh_window_clamp_test PRIVATE qalsh4c_qalsh)
    target_include_directories(qalsh_window_clamp_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    if(QALSH_WINDOW_FORCE_SCALAR)
        target_compile_definitions(qalsh_window_clamp_test PRIVATE QALSH_WINDOW_FORCE_SCALAR)
    endif()
    add_test(NAME qalsh_window_clamp_test COMMAND qalsh_window_clamp_test)

    add_executable(qalsh_window_clamp_scalar_test tests/window_clamp_test.cc)
    target_link_libraries(qalsh_window_clamp_scalar_test PRIVATE qalsh4c_qalsh)
    target_include_directories(qalsh_window_clamp_scalar_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(qalsh_window_clamp_scalar_test PRIVATE QALSH_WINDOW_FORCE_SCALAR)
    add_test(NAME qalsh_window_clamp_scalar_test COMMAND qalsh_window_clamp_scalar_test)

    add_executable(qalsh_window_trace_test tests/window_trace_test.cc)
    target_link_libraries(qalsh_window_trace_test PRIVATE qalsh4c_qalsh)
    if(QALSH_WINDOW_FORCE_SCALAR)
        target_compile_definitions(qalsh_window_trace_test PRIVATE QALSH_WINDOW_FORCE_SCALAR)
    endif()
    add_test(NAME qalsh_window_trace_test COMMAND qalsh_window_trace_test)
endif()
