set(source_data_dir "${QALSH4C_SOURCE_DIR}/data/toy")

function(prepare_dataset dataset_dir)
    file(REMOVE_RECURSE "${dataset_dir}")
    file(MAKE_DIRECTORY "${dataset_dir}")
    foreach(file_name A.bin B.bin metadata.json)
        file(COPY "${source_data_dir}/${file_name}" DESTINATION "${dataset_dir}")
    endforeach()
endfunction()

function(check_index_exists dataset_dir point_set expected)
    set(config_path "${dataset_dir}/index/l2/${point_set}/config.json")
    if(expected)
        if(NOT EXISTS "${config_path}")
            message(FATAL_ERROR "Expected ${point_set} index at ${config_path}")
        endif()
    else()
        if(EXISTS "${config_path}")
            message(FATAL_ERROR "Did not expect ${point_set} index at ${config_path}")
        endif()
    endif()
endfunction()

function(run_index name direction expected_a expected_b)
    set(dataset_dir "${QALSH4C_BINARY_DIR}/direction_index_${name}")
    prepare_dataset("${dataset_dir}")

    set(command_args --use-fixed-seed index -p 2 -d "${dataset_dir}")
    if(NOT direction STREQUAL "default")
        list(APPEND command_args --direction "${direction}")
    endif()

    execute_process(
        COMMAND "${QALSH4C_BIN}" ${command_args}
        RESULT_VARIABLE index_result
        OUTPUT_VARIABLE index_stdout
        ERROR_VARIABLE index_stderr
    )
    if(NOT index_result EQUAL 0)
        message(FATAL_ERROR "${name} index failed with exit code ${index_result}\n${index_stdout}\n${index_stderr}")
    endif()

    check_index_exists("${dataset_dir}" "A" ${expected_a})
    check_index_exists("${dataset_dir}" "B" ${expected_b})
endfunction()

function(check_missing_reverse_index)
    set(dataset_dir "${QALSH4C_BINARY_DIR}/direction_index_missing_reverse")
    prepare_dataset("${dataset_dir}")

    execute_process(
        COMMAND "${QALSH4C_BIN}" --use-fixed-seed index -p 2 -d "${dataset_dir}" --direction ab
        RESULT_VARIABLE index_result
        OUTPUT_VARIABLE index_stdout
        ERROR_VARIABLE index_stderr
    )
    if(NOT index_result EQUAL 0)
        message(FATAL_ERROR "Preparing missing reverse index fixture failed with exit code ${index_result}\n${index_stdout}\n${index_stderr}")
    endif()

    execute_process(
        COMMAND "${QALSH4C_BIN}" estimate -p 2 -d "${dataset_dir}" --direction ba ann qalsh
        RESULT_VARIABLE estimate_result
        OUTPUT_VARIABLE estimate_stdout
        ERROR_VARIABLE estimate_stderr
    )
    if(estimate_result EQUAL 0)
        message(FATAL_ERROR "Expected missing reverse index estimate to fail\n${estimate_stdout}\n${estimate_stderr}")
    endif()

    string(CONCAT estimate_output "${estimate_stdout}" "${estimate_stderr}")
    if(NOT estimate_output MATCHES "Missing index for point set A")
        message(FATAL_ERROR "Expected missing index message for point set A:\n${estimate_output}")
    endif()
    if(NOT estimate_output MATCHES "--direction ba")
        message(FATAL_ERROR "Expected suggested ba indexing command:\n${estimate_output}")
    endif()
endfunction()

run_index("default" "default" FALSE TRUE)
run_index("ab" "ab" FALSE TRUE)
run_index("ba" "ba" TRUE FALSE)
run_index("both" "both" TRUE TRUE)
check_missing_reverse_index()
