set(source_data_dir "${QALSH4C_SOURCE_DIR}/data/federalist")
set(dataset_dir "${QALSH4C_BINARY_DIR}/in_memory_qalsh")

file(REMOVE_RECURSE "${dataset_dir}")
file(MAKE_DIRECTORY "${dataset_dir}")
foreach(file_name A.bin B.bin metadata.json)
    file(COPY "${source_data_dir}/${file_name}" DESTINATION "${dataset_dir}")
endforeach()

execute_process(
    COMMAND "${QALSH4C_BIN}" --help
    RESULT_VARIABLE help_result
    OUTPUT_VARIABLE help_stdout
    ERROR_VARIABLE help_stderr
)
if(NOT help_result EQUAL 0)
    message(FATAL_ERROR "Running qalsh4c --help failed with exit code ${help_result}")
endif()
string(CONCAT help_output "${help_stdout}" "${help_stderr}")
if(help_output MATCHES "[\r\n][ \t]+index[ \t\r\n]")
    message(FATAL_ERROR "Top-level help still lists removed index command:\n${help_output}")
endif()

execute_process(
    COMMAND "${QALSH4C_BIN}" index
    RESULT_VARIABLE index_result
    OUTPUT_VARIABLE index_stdout
    ERROR_VARIABLE index_stderr
)
if(index_result EQUAL 0)
    message(FATAL_ERROR "Expected removed index command to fail\n${index_stdout}\n${index_stderr}")
endif()

execute_process(
    COMMAND "${QALSH4C_BIN}" --use-fixed-seed estimate -p 2 -d "${dataset_dir}" --direction ab ann qalsh -c 2.2
    RESULT_VARIABLE ab_result
    OUTPUT_VARIABLE ab_stdout
    ERROR_VARIABLE ab_stderr
)
if(NOT ab_result EQUAL 0)
    message(FATAL_ERROR "In-memory QALSH A->B estimate failed with exit code ${ab_result}\n${ab_stdout}\n${ab_stderr}")
endif()
if(NOT ab_stdout MATCHES "Estimated Distance:")
    message(FATAL_ERROR "In-memory QALSH A->B output did not include an estimate:\n${ab_stdout}")
endif()
if(EXISTS "${dataset_dir}/index")
    message(FATAL_ERROR "In-memory QALSH estimate created an index directory at ${dataset_dir}/index")
endif()

execute_process(
    COMMAND "${QALSH4C_BIN}" --use-fixed-seed estimate -p 2 -d "${dataset_dir}" --direction both ann qalsh
    RESULT_VARIABLE both_result
    OUTPUT_VARIABLE both_stdout
    ERROR_VARIABLE both_stderr
)
if(NOT both_result EQUAL 0)
    message(FATAL_ERROR "In-memory QALSH bidirectional estimate failed with exit code ${both_result}\n${both_stdout}\n${both_stderr}")
endif()
if(NOT both_stdout MATCHES "Direction: A <-> B")
    message(FATAL_ERROR "In-memory QALSH bidirectional output did not report both directions:\n${both_stdout}")
endif()
if(EXISTS "${dataset_dir}/index")
    message(FATAL_ERROR "In-memory QALSH bidirectional estimate created an index directory at ${dataset_dir}/index")
endif()
