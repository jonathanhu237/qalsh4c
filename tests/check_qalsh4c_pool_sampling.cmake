set(source_data_dir "${QALSH4C_SOURCE_DIR}/data/federalist")
set(dataset_dir "${QALSH4C_BINARY_DIR}/pool_sampling")

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

function(run_pool_estimate norm)
    set(estimate_context "pool L${norm}")
    execute_process(
        COMMAND "${QALSH4C_BIN}" --use-fixed-seed estimate
            -p "${norm}"
            -d "${dataset_dir}"
            --direction ab
            pool
            -K 200
            -n 50
            -s 20
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
    check_contains("${estimate_stdout}" "Relative Error:" "${estimate_context}")
    check_contains("${estimate_stderr}" "Pool sampling stats:" "${estimate_context}")
endfunction()

execute_process(
    COMMAND "${QALSH4C_BIN}" estimate pool --help
    RESULT_VARIABLE help_result
    OUTPUT_VARIABLE help_stdout
    ERROR_VARIABLE help_stderr
)
if(NOT help_result EQUAL 0)
    message(FATAL_ERROR "Running pool help failed with exit code ${help_result}")
endif()
string(CONCAT help_output "${help_stdout}" "${help_stderr}")
check_contains("${help_output}" "--pool-size" "pool help")
check_contains("${help_output}" "--subsample-size" "pool help")

run_pool_estimate(1)
run_pool_estimate(2)

execute_process(
    COMMAND "${QALSH4C_BIN}" index qalsh -p 2 -d "${dataset_dir}"
    RESULT_VARIABLE index_result
    OUTPUT_VARIABLE index_stdout
    ERROR_VARIABLE index_stderr
)
if(NOT index_result EQUAL 0)
    message(FATAL_ERROR "Building the disk QALSH index failed:\n${index_stdout}\n${index_stderr}")
endif()

# The index stores one 8-byte fingerprint per target point for exact-match detection.
set(fingerprints_path "${dataset_dir}/index/qalsh/l2/b/fingerprints.bin")
function(check_fingerprints context)
    if(NOT EXISTS "${fingerprints_path}")
        message(FATAL_ERROR "Expected ${fingerprints_path} after ${context}")
    endif()
    file(SIZE "${fingerprints_path}" fingerprints_size)
    if(NOT fingerprints_size EQUAL 9424)
        message(FATAL_ERROR "Expected 1178 fingerprints (9424 bytes) after ${context}, got ${fingerprints_size} bytes")
    endif()
endfunction()
check_fingerprints("building the index")

function(run_disk_pool_estimate out_estimate)
    execute_process(
        COMMAND "${QALSH4C_BIN}" --use-fixed-seed estimate --disk-based
            -p 2
            -d "${dataset_dir}"
            --direction ab
            pool
            -K 200
            -n 50
            -s 20
        RESULT_VARIABLE disk_result
        OUTPUT_VARIABLE disk_stdout
        ERROR_VARIABLE disk_stderr
    )
    if(NOT disk_result EQUAL 0)
        message(FATAL_ERROR "Disk-based pool estimate failed:\n${disk_stdout}\n${disk_stderr}")
    endif()
    check_contains("${disk_stdout}" "Relative Error:" "disk pool L2")
    check_contains("${disk_stderr}" "Pool sampling stats:" "disk pool L2")
    string(REGEX MATCH "Estimated Distance: [^\n]*" estimate_line "${disk_stdout}")
    set(${out_estimate} "${estimate_line}" PARENT_SCOPE)
endfunction()

# Stored fingerprints and the fallback pass over B must find the same exact matches.
run_disk_pool_estimate(with_fingerprints)
file(REMOVE "${fingerprints_path}")
run_disk_pool_estimate(without_fingerprints)
if(NOT with_fingerprints STREQUAL without_fingerprints)
    message(FATAL_ERROR "Disk pool estimate changed without stored fingerprints: "
        "'${with_fingerprints}' vs '${without_fingerprints}'")
endif()

# Fingerprints that no longer match the target points are ignored instead of trusted.
file(WRITE "${fingerprints_path}" "")
foreach(point_index RANGE 1177)
    file(APPEND "${fingerprints_path}" "00000000")
endforeach()
run_disk_pool_estimate(with_stale_fingerprints)
if(NOT with_stale_fingerprints STREQUAL without_fingerprints)
    message(FATAL_ERROR "Disk pool estimate trusted stale fingerprints: "
        "'${with_stale_fingerprints}' vs '${without_fingerprints}'")
endif()
file(REMOVE "${fingerprints_path}")

# Rerunning the index build on an existing index adds missing fingerprints.
execute_process(
    COMMAND "${QALSH4C_BIN}" index qalsh -p 2 -d "${dataset_dir}"
    RESULT_VARIABLE backfill_result
    OUTPUT_VARIABLE backfill_stdout
    ERROR_VARIABLE backfill_stderr
)
if(NOT backfill_result EQUAL 0)
    message(FATAL_ERROR "Re-running the disk QALSH index build failed:\n${backfill_stdout}\n${backfill_stderr}")
endif()
check_fingerprints("re-running the index build")
