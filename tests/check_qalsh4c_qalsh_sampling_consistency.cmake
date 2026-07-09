set(source_data_dir "${QALSH4C_SOURCE_DIR}/data/federalist")
set(work_dir "${QALSH4C_BINARY_DIR}/qalsh_sampling_consistency")
# In-memory QALSH draws its projection seed from the fixed mt19937(42) generator.
set(qalsh_seed "1608637542")

function(copy_dataset dataset_dir)
    file(REMOVE_RECURSE "${dataset_dir}")
    file(MAKE_DIRECTORY "${dataset_dir}")
    foreach(file_name A.bin B.bin metadata.json)
        file(COPY "${source_data_dir}/${file_name}" DESTINATION "${dataset_dir}")
    endforeach()
endfunction()

function(run_ok name)
    execute_process(
        COMMAND "${QALSH4C_BIN}" ${ARGN}
        RESULT_VARIABLE result
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr
    )
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "${name} failed with exit code ${result}\n${stdout}\n${stderr}")
    endif()
    string(CONCAT output "${stdout}" "${stderr}")
    set("${name}_output" "${output}" PARENT_SCOPE)
endfunction()

function(extract_estimate output_var output context)
    string(REGEX MATCH "Estimated Distance: ([0-9]+\\.[0-9]+)" estimate_match "${output}")
    if(NOT estimate_match)
        message(FATAL_ERROR "Could not find estimated distance in ${context} output:\n${output}")
    endif()
    set("${output_var}" "${CMAKE_MATCH_1}" PARENT_SCOPE)
endfunction()

function(check_case direction target weight_file)
    set(in_memory_dir "${work_dir}/l2_${direction}_in_memory")
    set(disk_dir "${work_dir}/l2_${direction}_disk")

    copy_dataset("${in_memory_dir}")
    copy_dataset("${disk_dir}")

    run_ok(in_memory
        --use-fixed-seed -T 1 estimate -p 2 -d "${in_memory_dir}" --direction "${direction}"
        sampling --use-cache -n 64 --exact-batch-size 1 qalsh -c 2
    )
    run_ok(index
        index qalsh -p 2 -d "${disk_dir}" --target "${target}" --seed "${qalsh_seed}" -c 2 --overwrite
    )
    run_ok(disk
        --use-fixed-seed -T 1 estimate --disk-based --disk-query-batch-size 37 --disk-block-size 257
        -p 2 -d "${disk_dir}" --direction "${direction}"
        sampling --use-cache -n 64 --exact-batch-size 1 qalsh -c 2
    )

    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E compare_files
            "${in_memory_dir}/${weight_file}"
            "${disk_dir}/${weight_file}"
        RESULT_VARIABLE compare_result
    )
    if(NOT compare_result EQUAL 0)
        message(FATAL_ERROR "QALSH sampling weights differ for direction ${direction}")
    endif()

    extract_estimate(in_memory_estimate "${in_memory_output}" "in-memory ${direction}")
    extract_estimate(disk_estimate "${disk_output}" "disk ${direction}")
    if(NOT in_memory_estimate STREQUAL disk_estimate)
        message(FATAL_ERROR
            "QALSH sampling estimate differs for direction ${direction}: "
            "in-memory=${in_memory_estimate}, disk=${disk_estimate}"
        )
    endif()
endfunction()

file(REMOVE_RECURSE "${work_dir}")
check_case(ab b l2_qalsh_weights_A.bin)
check_case(ba a l2_qalsh_weights_B.bin)
