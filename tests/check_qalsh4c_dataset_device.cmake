function(check_help_contains_device command_name)
    execute_process(
        COMMAND "${QALSH4C_BIN}" ${ARGN} --help
        RESULT_VARIABLE help_result
        OUTPUT_VARIABLE help_stdout
        ERROR_VARIABLE help_stderr
    )
    if(NOT help_result EQUAL 0)
        message(FATAL_ERROR "Running ${command_name} --help failed\n${help_stdout}\n${help_stderr}")
    endif()
    string(CONCAT help_output "${help_stdout}" "${help_stderr}")
    if(NOT help_output MATCHES "--ground-truth-device")
        message(FATAL_ERROR "Expected --ground-truth-device in ${command_name} help\n${help_output}")
    endif()
    if(NOT help_output MATCHES "--cuda-batch-size")
        message(FATAL_ERROR "Expected --cuda-batch-size in ${command_name} help\n${help_output}")
    endif()
endfunction()

check_help_contains_device("dataset generate gmm" dataset generate gmm)
check_help_contains_device("dataset convert p53" dataset convert p53)
check_help_contains_device("dataset ground-truth" dataset ground-truth)

set(cpu_dataset_dir "${QALSH4C_BINARY_DIR}/dataset_device_cpu")
set(auto_dataset_dir "${QALSH4C_BINARY_DIR}/dataset_device_auto")
set(cuda_dataset_dir "${QALSH4C_BINARY_DIR}/dataset_device_cuda")
file(REMOVE_RECURSE "${cpu_dataset_dir}" "${auto_dataset_dir}" "${cuda_dataset_dir}")

execute_process(
    COMMAND "${QALSH4C_BIN}" dataset generate gmm
        --num-points-a 2
        --num-points-b 2
        --num-dimensions 2
        --seed 21
        --ground-truth l2:ab
        --ground-truth-device cpu
        --no-progress
        -o "${cpu_dataset_dir}"
    RESULT_VARIABLE cpu_result
    OUTPUT_VARIABLE cpu_stdout
    ERROR_VARIABLE cpu_stderr
)
if(NOT cpu_result EQUAL 0)
    message(FATAL_ERROR "Expected CPU ground-truth device to succeed\n${cpu_stdout}\n${cpu_stderr}")
endif()

execute_process(
    COMMAND "${QALSH4C_BIN}" estimate -p 2 -d "${cpu_dataset_dir}" --direction ab ann linear_scan
    RESULT_VARIABLE estimate_result
    OUTPUT_VARIABLE estimate_stdout
    ERROR_VARIABLE estimate_stderr
)
if(NOT estimate_result EQUAL 0 OR NOT estimate_stdout MATCHES "Relative Error: 0.00%")
    message(FATAL_ERROR "Expected CPU-selected ground truth to match exact estimate\n${estimate_stdout}\n${estimate_stderr}")
endif()

execute_process(
    COMMAND "${QALSH4C_BIN}" dataset generate gmm
        --num-points-a 2
        --num-points-b 2
        --num-dimensions 2
        --seed 22
        --ground-truth l2:ab
        --ground-truth-device gpu
        --no-progress
        -o "${cpu_dataset_dir}"
    RESULT_VARIABLE invalid_result
    OUTPUT_VARIABLE invalid_stdout
    ERROR_VARIABLE invalid_stderr
)
if(invalid_result EQUAL 0)
    message(FATAL_ERROR "Expected invalid ground-truth device to fail\n${invalid_stdout}\n${invalid_stderr}")
endif()

if(NOT QALSH4C_CUDA_ENABLED)
    execute_process(
        COMMAND "${QALSH4C_BIN}" dataset generate gmm
            --num-points-a 2
            --num-points-b 2
            --num-dimensions 2
            --seed 23
            --ground-truth l2:ab
            --ground-truth-device auto
            --no-progress
            -o "${auto_dataset_dir}"
        RESULT_VARIABLE auto_result
        OUTPUT_VARIABLE auto_stdout
        ERROR_VARIABLE auto_stderr
    )
    if(NOT auto_result EQUAL 0)
        message(FATAL_ERROR "Expected non-CUDA auto mode to fall back to CPU\n${auto_stdout}\n${auto_stderr}")
    endif()
    if(NOT auto_stderr MATCHES "CUDA support is not compiled in")
        message(FATAL_ERROR "Expected non-CUDA auto fallback warning\n${auto_stdout}\n${auto_stderr}")
    endif()

    execute_process(
        COMMAND "${QALSH4C_BIN}" dataset generate gmm
            --num-points-a 2
            --num-points-b 2
            --num-dimensions 2
            --seed 24
            --ground-truth l2:ab
            --ground-truth-device cuda
            --no-progress
            -o "${cuda_dataset_dir}"
        RESULT_VARIABLE cuda_result
        OUTPUT_VARIABLE cuda_stdout
        ERROR_VARIABLE cuda_stderr
    )
    if(cuda_result EQUAL 0)
        message(FATAL_ERROR "Expected required CUDA mode to fail in non-CUDA build\n${cuda_stdout}\n${cuda_stderr}")
    endif()
    if(NOT cuda_stderr MATCHES "CUDA support is not compiled in")
        message(FATAL_ERROR "Expected non-CUDA required CUDA error\n${cuda_stdout}\n${cuda_stderr}")
    endif()
endif()
