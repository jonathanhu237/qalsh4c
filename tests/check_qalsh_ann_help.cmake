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
