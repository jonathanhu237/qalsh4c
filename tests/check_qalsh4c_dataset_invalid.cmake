set(dataset_dir "${QALSH4C_BINARY_DIR}/dataset_invalid_cli")
file(REMOVE_RECURSE "${dataset_dir}")

execute_process(
    COMMAND "${QALSH4C_BIN}" dataset generate gmm
        --num-points-a 2
        --num-points-b 2
        --num-dimensions 2
        --seed 14
        --ground-truth l2:both
        --no-progress
        -o "${dataset_dir}"
    RESULT_VARIABLE invalid_result
    OUTPUT_VARIABLE invalid_stdout
    ERROR_VARIABLE invalid_stderr
)
if(invalid_result EQUAL 0)
    message(FATAL_ERROR "Expected invalid ground-truth selector to fail\n${invalid_stdout}\n${invalid_stderr}")
endif()
if(EXISTS "${dataset_dir}/A.bin" OR EXISTS "${dataset_dir}/B.bin" OR EXISTS "${dataset_dir}/metadata.json")
    message(FATAL_ERROR "Invalid ground-truth selector left partial dataset output")
endif()
if(NOT invalid_stderr MATCHES "Unsupported ground-truth selector")
    message(FATAL_ERROR "Expected unsupported selector error\n${invalid_stdout}\n${invalid_stderr}")
endif()

execute_process(
    COMMAND "${QALSH4C_BIN}" dataset ground-truth
        -d "${QALSH4C_SOURCE_DIR}/data/federalist"
        --ground-truth ","
        --no-progress
    RESULT_VARIABLE empty_result
    OUTPUT_VARIABLE empty_stdout
    ERROR_VARIABLE empty_stderr
)
if(empty_result EQUAL 0)
    message(FATAL_ERROR "Expected empty ground-truth selector to fail\n${empty_stdout}\n${empty_stderr}")
endif()
if(NOT empty_stderr MATCHES "Ground-truth selectors must be non-empty")
    message(FATAL_ERROR "Expected empty selector error\n${empty_stdout}\n${empty_stderr}")
endif()

execute_process(
    COMMAND "${QALSH4C_BIN}" dataset generate gmm
        --num-points-a 2
        --num-points-b 2
        --num-dimensions 2
        --seed 14
        --ground-truth l2:ab,
        --no-progress
        -o "${dataset_dir}"
    RESULT_VARIABLE trailing_result
    OUTPUT_VARIABLE trailing_stdout
    ERROR_VARIABLE trailing_stderr
)
if(trailing_result EQUAL 0)
    message(FATAL_ERROR "Expected trailing empty ground-truth selector to fail\n${trailing_stdout}\n${trailing_stderr}")
endif()
if(NOT trailing_stderr MATCHES "Ground-truth selectors must be non-empty")
    message(FATAL_ERROR "Expected trailing empty selector error\n${trailing_stdout}\n${trailing_stderr}")
endif()
