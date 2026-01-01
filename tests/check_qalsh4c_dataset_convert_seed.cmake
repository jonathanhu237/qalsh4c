set(raw_dir "${QALSH4C_BINARY_DIR}/dataset_convert_seed_raw")
set(dataset_dir_a "${QALSH4C_BINARY_DIR}/dataset_convert_seed_a")
set(dataset_dir_b "${QALSH4C_BINARY_DIR}/dataset_convert_seed_b")
file(REMOVE_RECURSE "${raw_dir}" "${dataset_dir_a}" "${dataset_dir_b}")
file(MAKE_DIRECTORY "${raw_dir}")
file(WRITE "${raw_dir}/K9.data" "1,2,3,4\n5,6,7,8\n9,10,11,12\n13,14,15,16\n")

foreach(output_dir IN ITEMS "${dataset_dir_a}" "${dataset_dir_b}")
    execute_process(
        COMMAND "${QALSH4C_BIN}" dataset convert p53
            --raw-dir "${raw_dir}"
            --num-points-a 2
            --seed 99
            --no-progress
            -o "${output_dir}"
        RESULT_VARIABLE convert_result
        OUTPUT_VARIABLE convert_stdout
        ERROR_VARIABLE convert_stderr
    )
    if(NOT convert_result EQUAL 0)
        message(FATAL_ERROR "dataset convert p53 failed\n${convert_stdout}\n${convert_stderr}")
    endif()
endforeach()

execute_process(COMMAND "${CMAKE_COMMAND}" -E compare_files "${dataset_dir_a}/A.bin" "${dataset_dir_b}/A.bin"
    RESULT_VARIABLE compare_a_result)
if(NOT compare_a_result EQUAL 0)
    message(FATAL_ERROR "Expected seeded conversion A.bin output to be reproducible")
endif()

execute_process(COMMAND "${CMAKE_COMMAND}" -E compare_files "${dataset_dir_a}/B.bin" "${dataset_dir_b}/B.bin"
    RESULT_VARIABLE compare_b_result)
if(NOT compare_b_result EQUAL 0)
    message(FATAL_ERROR "Expected seeded conversion B.bin output to be reproducible")
endif()
