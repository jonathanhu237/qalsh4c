set(source_data_dir "${QALSH4C_SOURCE_DIR}/data/federalist")
set(dataset_dir "${QALSH4C_BINARY_DIR}/qalsh_weights_cache")
set(cache_path "${dataset_dir}/l2_qalsh_weights_A.bin")

file(REMOVE_RECURSE "${dataset_dir}")
file(MAKE_DIRECTORY "${dataset_dir}")
foreach(file_name A.bin B.bin metadata.json)
    file(COPY "${source_data_dir}/${file_name}" DESTINATION "${dataset_dir}")
endforeach()

file(READ "${dataset_dir}/metadata.json" metadata)
string(JSON num_points_a GET "${metadata}" num_points a)
math(EXPR expected_cache_size "${num_points_a} * 8")

execute_process(
    COMMAND "${QALSH4C_BIN}" --use-fixed-seed estimate -p 2 -d "${dataset_dir}" --direction ab
            sampling --use-cache -n 4 qalsh -c 2
    RESULT_VARIABLE miss_result
    OUTPUT_VARIABLE miss_stdout
    ERROR_VARIABLE miss_stderr
)
if(NOT miss_result EQUAL 0)
    message(FATAL_ERROR "QALSH sampling cache miss run failed with exit code ${miss_result}\n${miss_stdout}\n${miss_stderr}")
endif()
if(NOT EXISTS "${cache_path}")
    message(FATAL_ERROR "QALSH sampling cache miss did not create ${cache_path}")
endif()
file(SIZE "${cache_path}" actual_cache_size)
if(NOT actual_cache_size EQUAL expected_cache_size)
    message(FATAL_ERROR "QALSH cache size mismatch: expected ${expected_cache_size}, got ${actual_cache_size}")
endif()

execute_process(
    COMMAND "${QALSH4C_BIN}" --use-fixed-seed estimate -p 2 -d "${dataset_dir}" --direction ab
            sampling --use-cache -n 4 qalsh -c 2
    RESULT_VARIABLE hit_result
    OUTPUT_VARIABLE hit_stdout
    ERROR_VARIABLE hit_stderr
)
if(NOT hit_result EQUAL 0)
    message(FATAL_ERROR "QALSH sampling cache hit run failed with exit code ${hit_result}\n${hit_stdout}\n${hit_stderr}")
endif()
string(CONCAT hit_output "${hit_stdout}" "${hit_stderr}")
if(NOT hit_output MATCHES "weights will be read from the cache")
    message(FATAL_ERROR "QALSH sampling cache hit did not report cache read:\n${hit_output}")
endif()
if(hit_output MATCHES "Generating weights using QALSH")
    message(FATAL_ERROR "QALSH sampling cache hit unexpectedly regenerated QALSH weights:\n${hit_output}")
endif()

file(WRITE "${cache_path}" "bad")
execute_process(
    COMMAND "${QALSH4C_BIN}" --use-fixed-seed estimate -p 2 -d "${dataset_dir}" --direction ab
            sampling --use-cache -n 4 qalsh -c 2
    RESULT_VARIABLE short_result
    OUTPUT_VARIABLE short_stdout
    ERROR_VARIABLE short_stderr
)
if(short_result EQUAL 0)
    message(FATAL_ERROR "Expected short QALSH cache file to fail\n${short_stdout}\n${short_stderr}")
endif()
string(CONCAT short_output "${short_stdout}" "${short_stderr}")
if(NOT short_output MATCHES "cache file is incomplete")
    message(FATAL_ERROR "Short QALSH cache failure did not report incomplete cache:\n${short_output}")
endif()
