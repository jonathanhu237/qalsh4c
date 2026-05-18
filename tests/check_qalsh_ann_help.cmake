execute_process(
    COMMAND "${QALSH4C_BIN}" estimate ann qalsh --help
    RESULT_VARIABLE qalsh_ann_help_result
    OUTPUT_VARIABLE qalsh_ann_help_stdout
    ERROR_VARIABLE qalsh_ann_help_stderr
)

if(NOT qalsh_ann_help_result EQUAL 0)
    message(FATAL_ERROR "Running qalsh ann --help failed with exit code ${qalsh_ann_help_result}")
endif()

string(CONCAT qalsh_ann_help_output "${qalsh_ann_help_stdout}" "${qalsh_ann_help_stderr}")

if(NOT qalsh_ann_help_output MATCHES "--candidate-limit")
    message(FATAL_ERROR "Expected --candidate-limit option in qalsh ann help output")
endif()

execute_process(
    COMMAND "${QALSH4C_BIN}" estimate sampling qalsh --help
    RESULT_VARIABLE qalsh_sampling_help_result
    OUTPUT_VARIABLE qalsh_sampling_help_stdout
    ERROR_VARIABLE qalsh_sampling_help_stderr
)

if(NOT qalsh_sampling_help_result EQUAL 0)
    message(FATAL_ERROR "Running qalsh sampling --help failed with exit code ${qalsh_sampling_help_result}")
endif()

string(CONCAT qalsh_sampling_help_output "${qalsh_sampling_help_stdout}" "${qalsh_sampling_help_stderr}")

foreach(expected_option --approximation-ratio --save-weights --load-weights)
    if(NOT qalsh_sampling_help_output MATCHES "${expected_option}")
        message(FATAL_ERROR "Expected ${expected_option} option in qalsh sampling help output")
    endif()
endforeach()
