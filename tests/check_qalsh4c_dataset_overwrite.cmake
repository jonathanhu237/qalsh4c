set(dataset_dir "${QALSH4C_BINARY_DIR}/dataset_overwrite_cli")
file(REMOVE_RECURSE "${dataset_dir}")

execute_process(
    COMMAND "${QALSH4C_BIN}" dataset generate gmm
        --num-points-a 2
        --num-points-b 2
        --num-dimensions 2
        --seed 13
        --no-progress
        -o "${dataset_dir}"
    RESULT_VARIABLE generate_result
    OUTPUT_VARIABLE generate_stdout
    ERROR_VARIABLE generate_stderr
)
if(NOT generate_result EQUAL 0)
    message(FATAL_ERROR "initial dataset generate failed\n${generate_stdout}\n${generate_stderr}")
endif()

file(MAKE_DIRECTORY "${dataset_dir}/index/l2/B")
file(WRITE "${dataset_dir}/index/l2/B/config.json" "{}")

execute_process(
    COMMAND "${QALSH4C_BIN}" dataset generate gmm
        --num-points-a 2
        --num-points-b 2
        --num-dimensions 2
        --seed 13
        --no-progress
        -o "${dataset_dir}"
    RESULT_VARIABLE repeat_result
    OUTPUT_VARIABLE repeat_stdout
    ERROR_VARIABLE repeat_stderr
)
if(repeat_result EQUAL 0)
    message(FATAL_ERROR "Expected dataset generate without overwrite to fail\n${repeat_stdout}\n${repeat_stderr}")
endif()

execute_process(
    COMMAND "${QALSH4C_BIN}" dataset generate gmm
        --num-points-a 2
        --num-points-b 2
        --num-dimensions 2
        --seed 13
        --overwrite
        --no-progress
        -o "${dataset_dir}"
    RESULT_VARIABLE overwrite_result
    OUTPUT_VARIABLE overwrite_stdout
    ERROR_VARIABLE overwrite_stderr
)
if(NOT overwrite_result EQUAL 0)
    message(FATAL_ERROR "Expected dataset generate overwrite to succeed\n${overwrite_stdout}\n${overwrite_stderr}")
endif()
if(EXISTS "${dataset_dir}/index")
    message(FATAL_ERROR "Expected overwrite to remove stale index directory")
endif()
