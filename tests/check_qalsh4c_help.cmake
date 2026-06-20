execute_process(
    COMMAND "${QALSH4C_BIN}" --help
    RESULT_VARIABLE qalsh4c_help_result
    OUTPUT_VARIABLE qalsh4c_help_stdout
    ERROR_VARIABLE qalsh4c_help_stderr
)

if(NOT qalsh4c_help_result EQUAL 0)
    message(FATAL_ERROR "Running qalsh4c --help failed with exit code ${qalsh4c_help_result}")
endif()

string(CONCAT qalsh4c_help_output "${qalsh4c_help_stdout}" "${qalsh4c_help_stderr}")

if(qalsh4c_help_output MATCHES "--log-level")
    message(FATAL_ERROR "Unexpected --log-level option found in qalsh4c --help output")
endif()

if(NOT qalsh4c_help_output MATCHES "QALSH")
    message(FATAL_ERROR "Unexpected qalsh4c --help output: missing expected application text")
endif()

if(NOT qalsh4c_help_output MATCHES "[\r\n][ \t]+index[ \t\r\n]")
    message(FATAL_ERROR "Expected index command in qalsh4c --help output:\n${qalsh4c_help_output}")
endif()
