# The estimate and sampling help pages list the options of the in-memory pipeline and nothing disk-based.
function(help_output output_var)
    execute_process(
        COMMAND "${QALSH4C_BIN}" ${ARGN} --help
        RESULT_VARIABLE help_result
        OUTPUT_VARIABLE help_stdout
        ERROR_VARIABLE help_stderr
    )
    if(NOT help_result EQUAL 0)
        message(FATAL_ERROR "Running ${ARGN} --help failed with exit code ${help_result}")
    endif()
    string(CONCAT output "${help_stdout}" "${help_stderr}")
    set(${output_var} "${output}" PARENT_SCOPE)
endfunction()

help_output(estimate_help estimate)
if(NOT estimate_help MATCHES "--direction")
    message(FATAL_ERROR "Expected --direction in estimate --help:\n${estimate_help}")
endif()
if(estimate_help MATCHES "--disk")
    message(FATAL_ERROR "Unexpected disk option in estimate --help:\n${estimate_help}")
endif()

help_output(sampling_help estimate sampling)
foreach(option --repeats --exact-distances --num-samples)
    if(NOT sampling_help MATCHES "${option}")
        message(FATAL_ERROR "Expected ${option} in estimate sampling --help:\n${sampling_help}")
    endif()
endforeach()
if(NOT sampling_help MATCHES "precomputed")
    message(FATAL_ERROR "Expected the precomputed weights subcommand in estimate sampling --help:\n${sampling_help}")
endif()
