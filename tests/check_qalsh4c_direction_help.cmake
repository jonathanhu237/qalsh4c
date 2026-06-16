function(check_help command_name)
    execute_process(
        COMMAND "${QALSH4C_BIN}" "${command_name}" --help
        RESULT_VARIABLE help_result
        OUTPUT_VARIABLE help_stdout
        ERROR_VARIABLE help_stderr
    )

    if(NOT help_result EQUAL 0)
        message(FATAL_ERROR "Running ${command_name} --help failed with exit code ${help_result}")
    endif()

    string(CONCAT help_output "${help_stdout}" "${help_stderr}")
    if(NOT help_output MATCHES "--direction")
        message(FATAL_ERROR "Expected --direction option in ${command_name} help output")
    endif()
    if(NOT help_output MATCHES "--disk-based")
        message(FATAL_ERROR "Expected --disk-based option in ${command_name} help output")
    endif()
    if(NOT help_output MATCHES "--disk-block-size")
        message(FATAL_ERROR "Expected --disk-block-size option in ${command_name} help output")
    endif()
    if(NOT help_output MATCHES "--disk-query-batch-size")
        message(FATAL_ERROR "Expected --disk-query-batch-size option in ${command_name} help output")
    endif()
endfunction()

check_help(estimate)
