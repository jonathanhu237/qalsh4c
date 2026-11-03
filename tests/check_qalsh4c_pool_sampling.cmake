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

function(run_static_estimate norm)
    set(estimate_context "pool --static L${norm}")
    execute_process(
        COMMAND "${QALSH4C_BIN}" --use-fixed-seed estimate
            -p "${norm}"
            -d "${dataset_dir}"
            --direction ab
            pool
            --static
            --static-repetitions 2
            --static-window 2
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

    check_contains("${estimate_stdout}" "Estimated Distance:" "${estimate_context}")
    check_contains("${estimate_stdout}" "Relative Error:" "${estimate_context}")
    check_contains("${estimate_stderr}" "Static C2LSH stats: m=" "${estimate_context}")
    check_contains("${estimate_stderr}" "repetitions=2 window=2" "${estimate_context}")
endfunction()

check_contains("${help_output}" "--static-repetitions" "pool help")
run_static_estimate(1)
run_static_estimate(2)

# The static design replaces the direct baseline's bounds, so the two flags exclude each other.
execute_process(
    COMMAND "${QALSH4C_BIN}" estimate -p 2 -d "${dataset_dir}" --direction ab pool --static --direct -n 50
    RESULT_VARIABLE static_direct_result
    OUTPUT_VARIABLE static_direct_stdout
    ERROR_VARIABLE static_direct_stderr
)
if(static_direct_result EQUAL 0)
    message(FATAL_ERROR "pool --static --direct should fail:\n${static_direct_stdout}\n${static_direct_stderr}")
endif()
check_contains("${static_direct_stderr}" "do not combine it with" "pool --static --direct")

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

# External-memory static design: fixed T, and a model plan whose pilot scans B from disk.
function(run_disk_static_estimate context)
    execute_process(
        COMMAND "${QALSH4C_BIN}" --use-fixed-seed estimate --disk-based
            -p 2
            -d "${dataset_dir}"
            --direction ab
            pool
            --static
            --static-repetitions 2
            -s 20
            ${ARGN}
        RESULT_VARIABLE disk_result
        OUTPUT_VARIABLE disk_stdout
        ERROR_VARIABLE disk_stderr
    )
    if(NOT disk_result EQUAL 0)
        message(FATAL_ERROR "${context} failed:\n${disk_stdout}\n${disk_stderr}")
    endif()
    check_contains("${disk_stdout}" "Relative Error:" "${context}")
    # A broken estimate, such as zero from an empty pool, still prints a relative error; require a sane one.
    string(REGEX MATCH "Relative Error: ([0-9.]+)%" error_line "${disk_stdout}")
    if(NOT CMAKE_MATCH_1 LESS 30)
        message(FATAL_ERROR "${context} has relative error ${CMAKE_MATCH_1}%:\n${disk_stdout}")
    endif()
    check_contains("${disk_stderr}" "Static C2LSH stats: m=" "${context}")
    check_contains("${disk_stderr}" "verify_passes=1" "${context}")
    check_contains("${disk_stderr}" "Pool sampling stats:" "${context}")
endfunction()

run_disk_static_estimate("disk pool --static" -n 50)
run_disk_static_estimate("disk pool --static --plan model" --plan model --target-error 0.05)

# On disk a plan runs only the designs without QALSH.
execute_process(
    COMMAND "${QALSH4C_BIN}" estimate --disk-based -p 2 -d "${dataset_dir}" --direction ab
        pool --plan model --target-error 0.05
    RESULT_VARIABLE disk_plan_result
    OUTPUT_VARIABLE disk_plan_stdout
    ERROR_VARIABLE disk_plan_stderr
)
if(disk_plan_result EQUAL 0)
    message(FATAL_ERROR "A pooled plan on disk should fail:\n${disk_plan_stdout}\n${disk_plan_stderr}")
endif()
check_contains("${disk_plan_stderr}" "needs --direct or --static" "disk pooled plan")

# A persistent static index replaces the projection pass over B; with a plan, the pilot rides along with verification.
execute_process(
    COMMAND "${QALSH4C_BIN}" index static -p 2 -d "${dataset_dir}" --repetitions 2
    RESULT_VARIABLE static_index_result
    OUTPUT_VARIABLE static_index_stdout
    ERROR_VARIABLE static_index_stderr
)
if(NOT static_index_result EQUAL 0)
    message(FATAL_ERROR "Building the static index failed:\n${static_index_stdout}\n${static_index_stderr}")
endif()
foreach(index_file meta.json vectors.bin projections.bin)
    if(NOT EXISTS "${dataset_dir}/index/static/l2/b/${index_file}")
        message(FATAL_ERROR "Expected ${index_file} in the static index")
    endif()
endforeach()
run_disk_static_estimate("disk pool --static --static-index --plan model" --static-index --plan model --target-error 0.05)

function(expect_static_failure context expected)
    execute_process(
        COMMAND "${QALSH4C_BIN}" estimate ${ARGN}
        RESULT_VARIABLE failure_result
        OUTPUT_VARIABLE failure_stdout
        ERROR_VARIABLE failure_stderr
    )
    if(failure_result EQUAL 0)
        message(FATAL_ERROR "${context} should fail:\n${failure_stdout}\n${failure_stderr}")
    endif()
    check_contains("${failure_stderr}" "${expected}" "${context}")
endfunction()

expect_static_failure("an index with too few repetitions" "holds 2 repetitions"
    --disk-based -p 2 -d "${dataset_dir}" --direction ab pool --static --static-index --static-repetitions 3 -n 50)
expect_static_failure("--static-index in memory" "--static-index needs --disk-based"
    -p 2 -d "${dataset_dir}" --direction ab pool --static --static-index -n 50)
