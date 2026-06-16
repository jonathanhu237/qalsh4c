set(source_data_dir "${QALSH4C_SOURCE_DIR}/data/federalist")

function(check_contains output expected context)
    string(FIND "${output}" "${expected}" found_index)
    if(found_index EQUAL -1)
        message(FATAL_ERROR "Expected '${expected}' in ${context} output:\n${output}")
    endif()
endfunction()

function(read_ground_truth output_var norm direction)
    file(READ "${source_data_dir}/metadata.json" metadata)
    string(JSON expected GET "${metadata}" chamfer_distances "${norm}" "${direction}")
    set(${output_var} "${expected}" PARENT_SCOPE)
endfunction()

function(format_distance output_var value)
    string(REGEX MATCH "^[-+]?[0-9]+(\\.[0-9]+)?" numeric_value "${value}")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env EXPECTED_VALUE=${numeric_value}
                awk "BEGIN { printf \"%.6f\", ENVIRON[\"EXPECTED_VALUE\"] }"
        OUTPUT_VARIABLE formatted
        RESULT_VARIABLE format_result
    )
    if(NOT format_result EQUAL 0)
        message(FATAL_ERROR "Failed to format expected distance ${value}")
    endif()
    set(${output_var} "${formatted}" PARENT_SCOPE)
endfunction()

read_ground_truth(expected_l2_ab_raw l2 ab)
read_ground_truth(expected_l2_ba_raw l2 ba)
format_distance(expected_l2_ab "${expected_l2_ab_raw}")
format_distance(expected_l2_ba "${expected_l2_ba_raw}")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
            EXPECTED_AB=${expected_l2_ab_raw}
            EXPECTED_BA=${expected_l2_ba_raw}
            awk "BEGIN { printf \"%.6f\", ENVIRON[\"EXPECTED_AB\"] + ENVIRON[\"EXPECTED_BA\"] }"
    OUTPUT_VARIABLE expected_l2_both
    RESULT_VARIABLE both_format_result
)
if(NOT both_format_result EQUAL 0)
    message(FATAL_ERROR "Failed to format bidirectional expected distance")
endif()

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
    check_contains("${estimate_stdout}" "Estimated Distance:" "${name}")
    check_contains("${estimate_stdout}" "Ground Truth: ${expected_distance}" "${name}")
    check_contains("${estimate_stdout}" "Relative Error: 0.00%" "${name}")
endfunction()

run_estimate("default" "default" "A -> B" "${expected_l2_ab}")
run_estimate("ab" "ab" "A -> B" "${expected_l2_ab}")
run_estimate("ba" "ba" "B -> A" "${expected_l2_ba}")
run_estimate("both" "both" "A <-> B" "${expected_l2_both}")

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
