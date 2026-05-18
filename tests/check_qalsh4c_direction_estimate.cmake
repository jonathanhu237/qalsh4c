set(source_data_dir "${QALSH4C_SOURCE_DIR}/data/toy")

function(check_contains output expected context)
    string(FIND "${output}" "${expected}" found_index)
    if(found_index EQUAL -1)
        message(FATAL_ERROR "Expected '${expected}' in ${context} output:\n${output}")
    endif()
endfunction()

function(run_estimate name direction expected_direction expected_distance)
    set(command_args estimate -p 2 -d "${source_data_dir}")
    if(NOT direction STREQUAL "default")
        list(APPEND command_args --direction "${direction}")
    endif()
    list(APPEND command_args ann linear_scan)

    execute_process(
        COMMAND "${QALSH4C_BIN}" ${command_args}
        RESULT_VARIABLE estimate_result
        OUTPUT_VARIABLE estimate_stdout
        ERROR_VARIABLE estimate_stderr
    )
    if(NOT estimate_result EQUAL 0)
        message(FATAL_ERROR "${name} estimate failed with exit code ${estimate_result}\n${estimate_stdout}\n${estimate_stderr}")
    endif()

    check_contains("${estimate_stdout}" "Direction: ${expected_direction}" "${name}")
    check_contains("${estimate_stdout}" "Estimated Distance: ${expected_distance}" "${name}")
    check_contains("${estimate_stdout}" "Ground Truth: ${expected_distance}" "${name}")
    check_contains("${estimate_stdout}" "Relative Error: 0.00%" "${name}")
endfunction()

run_estimate("default" "default" "A -> B" "2.000000")
run_estimate("ab" "ab" "A -> B" "2.000000")
run_estimate("ba" "ba" "B -> A" "3.000000")
run_estimate("both" "both" "A <-> B" "5.000000")

execute_process(
    COMMAND "${QALSH4C_BIN}" --use-fixed-seed estimate -p 2 -d "${source_data_dir}" --direction both ann qalsh
    RESULT_VARIABLE qalsh_estimate_result
    OUTPUT_VARIABLE qalsh_estimate_stdout
    ERROR_VARIABLE qalsh_estimate_stderr
)
if(NOT qalsh_estimate_result EQUAL 0)
    message(FATAL_ERROR "QALSH estimate without prebuilt index failed with exit code ${qalsh_estimate_result}\n${qalsh_estimate_stdout}\n${qalsh_estimate_stderr}")
endif()
string(CONCAT qalsh_estimate_output "${qalsh_estimate_stdout}" "${qalsh_estimate_stderr}")
check_contains("${qalsh_estimate_output}" "Building in-memory QALSH index" "qalsh-online")
check_contains("${qalsh_estimate_stdout}" "Direction: A <-> B" "qalsh-online")
check_contains("${qalsh_estimate_stdout}" "Estimated Distance: 5.000000" "qalsh-online")

set(zero_data_dir "${QALSH4C_BINARY_DIR}/direction_zero_ground_truth")
set(zero_raw_dir "${QALSH4C_BINARY_DIR}/direction_zero_ground_truth_raw")
file(REMOVE_RECURSE "${zero_data_dir}")
file(REMOVE_RECURSE "${zero_raw_dir}")
file(MAKE_DIRECTORY "${zero_raw_dir}")
file(WRITE "${zero_raw_dir}/K9.data" "0,0,1,2\n0,0,3,4\n")
execute_process(
    COMMAND "${QALSH4C_BIN}" dataset convert p53
        --raw-dir "${zero_raw_dir}"
        --num-points-a 1
        --no-shuffle
        --ground-truth l2:ab
        --no-progress
        -o "${zero_data_dir}"
    RESULT_VARIABLE zero_data_result
    OUTPUT_VARIABLE zero_data_stdout
    ERROR_VARIABLE zero_data_stderr
)
if(NOT zero_data_result EQUAL 0)
    message(FATAL_ERROR "Failed to create zero-ground-truth dataset\n${zero_data_stdout}\n${zero_data_stderr}")
endif()

execute_process(
    COMMAND "${QALSH4C_BIN}" estimate -p 2 -d "${zero_data_dir}" --direction ab ann linear_scan
    RESULT_VARIABLE zero_estimate_result
    OUTPUT_VARIABLE zero_estimate_stdout
    ERROR_VARIABLE zero_estimate_stderr
)
if(NOT zero_estimate_result EQUAL 0)
    message(FATAL_ERROR "Zero-ground-truth estimate failed with exit code ${zero_estimate_result}\n${zero_estimate_stdout}\n${zero_estimate_stderr}")
endif()

check_contains("${zero_estimate_stdout}" "Estimated Distance: 0.000000" "zero-ground-truth")
check_contains("${zero_estimate_stdout}" "Ground Truth: 0.000000" "zero-ground-truth")
check_contains("${zero_estimate_stdout}" "Relative Error: 0.00%" "zero-ground-truth")
