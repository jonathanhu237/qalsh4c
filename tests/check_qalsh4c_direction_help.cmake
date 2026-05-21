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
endfunction()

check_help(index)
check_help(estimate)
