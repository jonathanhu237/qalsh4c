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

set(zero_data_dir "${QALSH4C_BINARY_DIR}/direction_zero_ground_truth")
file(REMOVE_RECURSE "${zero_data_dir}")
file(MAKE_DIRECTORY "${zero_data_dir}")
execute_process(
    COMMAND python3 -c "from pathlib import Path; import json, struct; d=Path('${zero_data_dir}'); data=struct.pack('2d', 0.0, 0.0); (d/'A.bin').write_bytes(data); (d/'B.bin').write_bytes(data); (d/'metadata.json').write_text(json.dumps({'num_dimensions': 2, 'num_points': {'a': 1, 'b': 1}, 'chamfer_distances': {'l1': {'ab': 0.0, 'ba': 0.0}, 'l2': {'ab': 0.0, 'ba': 0.0}}}, indent=4))"
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
