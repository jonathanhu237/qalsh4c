set(source_data_dir "${QALSH4C_SOURCE_DIR}/data/gmm")
set(weights_dir "${QALSH4C_BINARY_DIR}/qalsh_sampling_weights")

file(REMOVE_RECURSE "${weights_dir}")

execute_process(
    COMMAND "${QALSH4C_BIN}" --use-fixed-seed estimate -p 2 -d "${source_data_dir}" --direction ab
        sampling -n 3 qalsh --save-weights "${weights_dir}"
    RESULT_VARIABLE save_result
    OUTPUT_VARIABLE save_stdout
    ERROR_VARIABLE save_stderr
)
if(NOT save_result EQUAL 0)
    message(FATAL_ERROR "Saving QALSH weights failed with exit code ${save_result}\n${save_stdout}\n${save_stderr}")
endif()
if(NOT save_stdout MATCHES "Estimated Distance: ([0-9]+\\.[0-9]+)")
    message(FATAL_ERROR "Expected saved weights run to print an estimated distance:\n${save_stdout}")
endif()
set(saved_estimate "${CMAKE_MATCH_1}")

if(NOT EXISTS "${weights_dir}/metadata.json")
    message(FATAL_ERROR "Expected QALSH weights metadata at ${weights_dir}/metadata.json")
endif()
if(NOT EXISTS "${weights_dir}/weights.bin")
    message(FATAL_ERROR "Expected QALSH weights binary at ${weights_dir}/weights.bin")
endif()

file(READ "${weights_dir}/metadata.json" metadata)
if(NOT metadata MATCHES "qalsh4c-qalsh-weights-v1")
    message(FATAL_ERROR "Expected QALSH weights format marker in metadata:\n${metadata}")
endif()
if(NOT metadata MATCHES "\"direction\": \"ab\"")
    message(FATAL_ERROR "Expected ab direction in metadata:\n${metadata}")
endif()

execute_process(
    COMMAND "${QALSH4C_BIN}" --use-fixed-seed estimate -p 2 -d "${source_data_dir}" --direction ab
        sampling -n 3 qalsh --load-weights "${weights_dir}"
    RESULT_VARIABLE load_result
    OUTPUT_VARIABLE load_stdout
    ERROR_VARIABLE load_stderr
)
if(NOT load_result EQUAL 0)
    message(FATAL_ERROR "Loading QALSH weights failed with exit code ${load_result}\n${load_stdout}\n${load_stderr}")
endif()
if(NOT load_stdout MATCHES "Estimated Distance: ${saved_estimate}")
    message(FATAL_ERROR "Expected loaded weights estimate to match saved weights estimate ${saved_estimate}:\n${load_stdout}")
endif()
if(load_stdout MATCHES "Building in-memory QALSH index")
    message(FATAL_ERROR "Loading QALSH weights should skip in-memory QALSH weight generation:\n${load_stdout}")
endif()

execute_process(
    COMMAND "${QALSH4C_BIN}" estimate -p 2 -d "${source_data_dir}" --direction ba
        sampling -n 3 qalsh --load-weights "${weights_dir}"
    RESULT_VARIABLE mismatch_result
    OUTPUT_VARIABLE mismatch_stdout
    ERROR_VARIABLE mismatch_stderr
)
if(mismatch_result EQUAL 0)
    message(FATAL_ERROR "Expected direction mismatch to fail when loading QALSH weights\n${mismatch_stdout}\n${mismatch_stderr}")
endif()
string(CONCAT mismatch_output "${mismatch_stdout}" "${mismatch_stderr}")
if(NOT mismatch_output MATCHES "metadata mismatch")
    message(FATAL_ERROR "Expected metadata mismatch message for wrong direction:\n${mismatch_output}")
endif()

execute_process(
    COMMAND "${QALSH4C_BIN}" estimate -p 2 -d "${source_data_dir}" --direction ab
        sampling -n 3 qalsh --save-weights "${weights_dir}" --load-weights "${weights_dir}"
    RESULT_VARIABLE combined_result
    OUTPUT_VARIABLE combined_stdout
    ERROR_VARIABLE combined_stderr
)
if(combined_result EQUAL 0)
    message(FATAL_ERROR "Expected combined save/load weights options to fail\n${combined_stdout}\n${combined_stderr}")
endif()
string(CONCAT combined_output "${combined_stdout}" "${combined_stderr}")
if(NOT combined_output MATCHES "excludes")
    message(FATAL_ERROR "Expected excludes error for combined save/load weights options:\n${combined_output}")
endif()

execute_process(
    COMMAND "${QALSH4C_BIN}" estimate -p 2 -d "${source_data_dir}" --direction both
        sampling -n 3 qalsh --save-weights "${weights_dir}"
    RESULT_VARIABLE both_save_result
    OUTPUT_VARIABLE both_save_stdout
    ERROR_VARIABLE both_save_stderr
)
if(both_save_result EQUAL 0)
    message(FATAL_ERROR "Expected bidirectional save weights usage to fail\n${both_save_stdout}\n${both_save_stderr}")
endif()
string(CONCAT both_save_output "${both_save_stdout}" "${both_save_stderr}")
if(NOT both_save_output MATCHES "only supported for --direction ab or --direction ba")
    message(FATAL_ERROR "Expected single-direction error for bidirectional save weights:\n${both_save_output}")
endif()

execute_process(
    COMMAND "${QALSH4C_BIN}" estimate -p 2 -d "${source_data_dir}" --direction both
        sampling -n 3 qalsh --load-weights "${weights_dir}"
    RESULT_VARIABLE both_load_result
    OUTPUT_VARIABLE both_load_stdout
    ERROR_VARIABLE both_load_stderr
)
if(both_load_result EQUAL 0)
    message(FATAL_ERROR "Expected bidirectional load weights usage to fail\n${both_load_stdout}\n${both_load_stderr}")
endif()
string(CONCAT both_load_output "${both_load_stdout}" "${both_load_stderr}")
if(NOT both_load_output MATCHES "only supported for --direction ab or --direction ba")
    message(FATAL_ERROR "Expected single-direction error for bidirectional load weights:\n${both_load_output}")
endif()
