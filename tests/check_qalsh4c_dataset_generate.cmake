set(dataset_dir "${QALSH4C_BINARY_DIR}/dataset_generate_cli")
set(dataset_dir_repro "${QALSH4C_BINARY_DIR}/dataset_generate_cli_repro")
file(REMOVE_RECURSE "${dataset_dir}")
file(REMOVE_RECURSE "${dataset_dir_repro}")

execute_process(
    COMMAND "${QALSH4C_BIN}" dataset generate gmm
        --num-points-a 2
        --num-points-b 2
        --num-dimensions 2
        --outlier-fraction 0.25
        --outlier-scale 4
        --seed 11
        --ground-truth l2:ab
        --no-progress
        -o "${dataset_dir}"
    RESULT_VARIABLE generate_result
    OUTPUT_VARIABLE generate_stdout
    ERROR_VARIABLE generate_stderr
)
if(NOT generate_result EQUAL 0)
    message(FATAL_ERROR "dataset generate failed\n${generate_stdout}\n${generate_stderr}")
endif()

foreach(file_name A.bin B.bin metadata.json)
    if(NOT EXISTS "${dataset_dir}/${file_name}")
        message(FATAL_ERROR "Expected generated file ${dataset_dir}/${file_name}")
    endif()
endforeach()

file(READ "${dataset_dir}/metadata.json" metadata)
if(NOT metadata MATCHES "\"ab\": null")
    message(FATAL_ERROR "Expected at least one null ab metadata value\n${metadata}")
endif()
if(NOT metadata MATCHES "\"l2\"")
    message(FATAL_ERROR "Expected l2 metadata\n${metadata}")
endif()

execute_process(
    COMMAND "${QALSH4C_BIN}" estimate -p 2 -d "${dataset_dir}" --direction ab ann linear_scan
    RESULT_VARIABLE estimate_l2_result
    OUTPUT_VARIABLE estimate_l2_stdout
    ERROR_VARIABLE estimate_l2_stderr
)
if(NOT estimate_l2_result EQUAL 0)
    message(FATAL_ERROR "l2 estimate failed\n${estimate_l2_stdout}\n${estimate_l2_stderr}")
endif()
if(NOT estimate_l2_stdout MATCHES "Ground Truth: [0-9]")
    message(FATAL_ERROR "Expected numeric l2 ground truth\n${estimate_l2_stdout}")
endif()

execute_process(
    COMMAND "${QALSH4C_BIN}" estimate -p 1 -d "${dataset_dir}" --direction ab ann linear_scan
    RESULT_VARIABLE estimate_l1_result
    OUTPUT_VARIABLE estimate_l1_stdout
    ERROR_VARIABLE estimate_l1_stderr
)
if(NOT estimate_l1_result EQUAL 0)
    message(FATAL_ERROR "l1 estimate failed\n${estimate_l1_stdout}\n${estimate_l1_stderr}")
endif()
if(NOT estimate_l1_stdout MATCHES "Ground Truth: unavailable")
    message(FATAL_ERROR "Expected unavailable l1 ground truth\n${estimate_l1_stdout}")
endif()
if(NOT estimate_l1_stdout MATCHES "Relative Error: unavailable")
    message(FATAL_ERROR "Expected unavailable l1 relative error\n${estimate_l1_stdout}")
endif()

execute_process(
    COMMAND "${QALSH4C_BIN}" dataset generate gmm
        --num-points-a 2
        --num-points-b 2
        --num-dimensions 2
        --outlier-fraction 0.25
        --outlier-scale 4
        --seed 11
        --ground-truth l2:ab
        --no-progress
        -o "${dataset_dir_repro}"
    RESULT_VARIABLE generate_repro_result
    OUTPUT_VARIABLE generate_repro_stdout
    ERROR_VARIABLE generate_repro_stderr
)
if(NOT generate_repro_result EQUAL 0)
    message(FATAL_ERROR "reproducible dataset generate failed\n${generate_repro_stdout}\n${generate_repro_stderr}")
endif()

execute_process(COMMAND "${CMAKE_COMMAND}" -E compare_files "${dataset_dir}/A.bin" "${dataset_dir_repro}/A.bin"
    RESULT_VARIABLE compare_a_result)
if(NOT compare_a_result EQUAL 0)
    message(FATAL_ERROR "Expected seeded GMM A.bin output to be reproducible")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -E compare_files "${dataset_dir}/B.bin" "${dataset_dir_repro}/B.bin"
    RESULT_VARIABLE compare_b_result)
if(NOT compare_b_result EQUAL 0)
    message(FATAL_ERROR "Expected seeded GMM B.bin output to be reproducible")
endif()
