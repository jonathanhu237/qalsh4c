set(source_data_dir "${QALSH4C_SOURCE_DIR}/data/federalist")
set(dataset_dir "${QALSH4C_BINARY_DIR}/quadtree_sampling")

file(REMOVE_RECURSE "${dataset_dir}")
file(MAKE_DIRECTORY "${dataset_dir}")
foreach(file_name A.bin B.bin metadata.json)
    file(COPY "${source_data_dir}/${file_name}" DESTINATION "${dataset_dir}")
endforeach()

function(check_contains output expected context)
    string(FIND "${output}" "${expected}" found_index)
    if(found_index EQUAL -1)
        message(FATAL_ERROR "Expected '${expected}' in ${context} output:\n${output}")
    endif()
endfunction()

function(run_quadtree_estimate norm max_levels)
    set(estimate_context "quadtree L${norm} max-levels ${max_levels}")
    execute_process(
        COMMAND "${QALSH4C_BIN}" --use-fixed-seed estimate
            -p "${norm}"
            -d "${dataset_dir}"
            --direction ab
            sampling
            -n 5
            quadtree
            --max-levels "${max_levels}"
        RESULT_VARIABLE estimate_result
        OUTPUT_VARIABLE estimate_stdout
        ERROR_VARIABLE estimate_stderr
    )
    if(NOT estimate_result EQUAL 0)
        message(FATAL_ERROR
            "${estimate_context} estimate failed with exit code ${estimate_result}\n"
            "${estimate_stdout}\n"
            "${estimate_stderr}"
        )
    endif()

    check_contains("${estimate_stdout}" "Direction: A -> B" "${estimate_context}")
    check_contains("${estimate_stdout}" "Estimated Distance:" "${estimate_context}")
    check_contains("${estimate_stdout}" "Ground Truth:" "${estimate_context}")
    check_contains("${estimate_stdout}" "Relative Error:" "${estimate_context}")
endfunction()

execute_process(
    COMMAND "${QALSH4C_BIN}" estimate sampling quadtree --help
    RESULT_VARIABLE help_result
    OUTPUT_VARIABLE help_stdout
    ERROR_VARIABLE help_stderr
)
if(NOT help_result EQUAL 0)
    message(FATAL_ERROR "Running quadtree sampling help failed with exit code ${help_result}")
endif()
string(CONCAT help_output "${help_stdout}" "${help_stderr}")
check_contains("${help_output}" "--max-levels" "quadtree help")
check_contains("${help_output}" "0 splits until singleton" "quadtree help")

run_quadtree_estimate(1 0)
run_quadtree_estimate(2 3)

execute_process(
    COMMAND "${QALSH4C_BIN}" --use-fixed-seed estimate
        -p 1
        -d "${dataset_dir}"
        --direction both
        sampling
        -n 3
        quadtree
        --max-levels 3
    RESULT_VARIABLE both_result
    OUTPUT_VARIABLE both_stdout
    ERROR_VARIABLE both_stderr
)
if(NOT both_result EQUAL 0)
    message(FATAL_ERROR "bidirectional quadtree estimate failed with exit code ${both_result}\n${both_stdout}\n${both_stderr}")
endif()
check_contains("${both_stdout}" "Direction: A <-> B" "bidirectional quadtree")

set(bad_cache_path "${dataset_dir}/l1_quadtree_l3_weights_A.bin")
file(WRITE "${bad_cache_path}" "bad-cache")
execute_process(
    COMMAND "${QALSH4C_BIN}" --use-fixed-seed estimate
        -p 1
        -d "${dataset_dir}"
        --direction ab
        sampling
        --use-cache
        -n 5
        quadtree
        --max-levels 3
    RESULT_VARIABLE bad_cache_result
    OUTPUT_VARIABLE bad_cache_stdout
    ERROR_VARIABLE bad_cache_stderr
)
if(bad_cache_result EQUAL 0)
    message(FATAL_ERROR "quadtree sampling unexpectedly accepted a malformed cache:\n${bad_cache_stdout}\n${bad_cache_stderr}")
endif()
string(CONCAT bad_cache_output "${bad_cache_stdout}" "${bad_cache_stderr}")
check_contains("${bad_cache_output}" "Quadtree weights cache file has unexpected size" "bad quadtree cache")
