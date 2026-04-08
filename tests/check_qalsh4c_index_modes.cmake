set(default_dir "${QALSH4C_BINARY_DIR}/index_mode_default")
set(fast_dir "${QALSH4C_BINARY_DIR}/index_mode_fast")
set(source_data_dir "${QALSH4C_SOURCE_DIR}/data/federalist")

file(REMOVE_RECURSE "${default_dir}" "${fast_dir}")
file(MAKE_DIRECTORY "${default_dir}" "${fast_dir}")

foreach(file_name A.bin B.bin metadata.json)
    file(COPY "${source_data_dir}/${file_name}" DESTINATION "${default_dir}")
    file(COPY "${source_data_dir}/${file_name}" DESTINATION "${fast_dir}")
endforeach()

execute_process(
    COMMAND "${QALSH4C_BIN}" --use-fixed-seed index -p 2 -d "${default_dir}"
    RESULT_VARIABLE default_result
    OUTPUT_VARIABLE default_stdout
    ERROR_VARIABLE default_stderr
)
if(NOT default_result EQUAL 0)
    message(FATAL_ERROR "Default indexing failed with exit code ${default_result}\n${default_stdout}\n${default_stderr}")
endif()

execute_process(
    COMMAND "${QALSH4C_BIN}" --use-fixed-seed index -p 2 -d "${fast_dir}" --fast
    RESULT_VARIABLE fast_result
    OUTPUT_VARIABLE fast_stdout
    ERROR_VARIABLE fast_stderr
)
if(NOT fast_result EQUAL 0)
    message(FATAL_ERROR "Fast indexing failed with exit code ${fast_result}\n${fast_stdout}\n${fast_stderr}")
endif()

file(GLOB_RECURSE default_index_files RELATIVE "${default_dir}" "${default_dir}/index/*")
file(GLOB_RECURSE fast_index_files RELATIVE "${fast_dir}" "${fast_dir}/index/*")

list(SORT default_index_files)
list(SORT fast_index_files)

if(NOT default_index_files STREQUAL fast_index_files)
    message(FATAL_ERROR "Default and fast indexing produced different file sets")
endif()

foreach(relative_path IN LISTS default_index_files)
    file(SHA256 "${default_dir}/${relative_path}" default_sha)
    file(SHA256 "${fast_dir}/${relative_path}" fast_sha)
    if(NOT default_sha STREQUAL fast_sha)
        message(FATAL_ERROR "Index file content mismatch for ${relative_path}")
    endif()
endforeach()
