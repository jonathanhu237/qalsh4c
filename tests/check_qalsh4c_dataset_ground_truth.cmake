set(dataset_dir "${QALSH4C_BINARY_DIR}/dataset_ground_truth_cli")
file(REMOVE_RECURSE "${dataset_dir}")

execute_process(
    COMMAND "${QALSH4C_BIN}" dataset generate gmm
        --num-points-a 2
        --num-points-b 2
        --num-dimensions 2
        --seed 12
        --no-progress
        -o "${dataset_dir}"
    RESULT_VARIABLE generate_result
    OUTPUT_VARIABLE generate_stdout
    ERROR_VARIABLE generate_stderr
)
if(NOT generate_result EQUAL 0)
    message(FATAL_ERROR "dataset generate failed\n${generate_stdout}\n${generate_stderr}")
endif()

execute_process(
    COMMAND "${QALSH4C_BIN}" dataset ground-truth
        -d "${dataset_dir}"
        --ground-truth l2:ab,l2:ab
        --no-progress
    RESULT_VARIABLE ground_truth_result
    OUTPUT_VARIABLE ground_truth_stdout
    ERROR_VARIABLE ground_truth_stderr
)
if(NOT ground_truth_result EQUAL 0)
    message(FATAL_ERROR "dataset ground-truth failed\n${ground_truth_stdout}\n${ground_truth_stderr}")
endif()

execute_process(
    COMMAND "${QALSH4C_BIN}" dataset ground-truth
        -d "${dataset_dir}"
        --ground-truth l2:ab
        --no-progress
    RESULT_VARIABLE repeat_result
    OUTPUT_VARIABLE repeat_stdout
    ERROR_VARIABLE repeat_stderr
)
if(repeat_result EQUAL 0)
    message(FATAL_ERROR "Expected repeat ground-truth without overwrite to fail\n${repeat_stdout}\n${repeat_stderr}")
endif()

execute_process(
    COMMAND "${QALSH4C_BIN}" dataset ground-truth
        -d "${dataset_dir}"
        --ground-truth l2:ab
        --overwrite
        --no-progress
    RESULT_VARIABLE overwrite_result
    OUTPUT_VARIABLE overwrite_stdout
    ERROR_VARIABLE overwrite_stderr
)
if(NOT overwrite_result EQUAL 0)
    message(FATAL_ERROR "Expected ground-truth overwrite to succeed\n${overwrite_stdout}\n${overwrite_stderr}")
endif()

execute_process(
    COMMAND "${QALSH4C_BIN}" estimate -p 2 -d "${dataset_dir}" --direction ab ann linear_scan
    RESULT_VARIABLE estimate_result
    OUTPUT_VARIABLE estimate_stdout
    ERROR_VARIABLE estimate_stderr
)
if(NOT estimate_result EQUAL 0)
    message(FATAL_ERROR "estimate failed after ground-truth\n${estimate_stdout}\n${estimate_stderr}")
endif()
if(NOT estimate_stdout MATCHES "Relative Error: 0.00%")
    message(FATAL_ERROR "Expected exact relative error after ground-truth\n${estimate_stdout}")
endif()
