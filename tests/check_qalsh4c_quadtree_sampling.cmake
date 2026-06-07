set(source_data_dir "${QALSH4C_SOURCE_DIR}/data/federalist")

function(check_contains output expected context)
    string(FIND "${output}" "${expected}" found_index)
    if(found_index EQUAL -1)
        message(FATAL_ERROR "Expected '${expected}' in ${context} output:\n${output}")
    endif()
endfunction()

function(run_quadtree_estimate norm)
    execute_process(
        COMMAND "${QALSH4C_BIN}" --use-fixed-seed estimate
            -p "${norm}"
            -d "${source_data_dir}"
            --direction ab
            sampling
            -n 5
            quadtree
            --max-levels 3
        RESULT_VARIABLE estimate_result
        OUTPUT_VARIABLE estimate_stdout
        ERROR_VARIABLE estimate_stderr
    )
    if(NOT estimate_result EQUAL 0)
        message(FATAL_ERROR "quadtree L${norm} estimate failed with exit code ${estimate_result}\n${estimate_stdout}\n${estimate_stderr}")
    endif()

    check_contains("${estimate_stdout}" "Direction: A -> B" "quadtree L${norm}")
    check_contains("${estimate_stdout}" "Estimated Distance:" "quadtree L${norm}")
    check_contains("${estimate_stdout}" "Ground Truth:" "quadtree L${norm}")
    check_contains("${estimate_stdout}" "Relative Error:" "quadtree L${norm}")
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

run_quadtree_estimate(1)
run_quadtree_estimate(2)
