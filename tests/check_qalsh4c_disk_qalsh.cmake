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

function(run_fail name)
    execute_process(
        COMMAND "${QALSH4C_BIN}" ${ARGN}
        RESULT_VARIABLE result
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr
    )
    if(result EQUAL 0)
        message(FATAL_ERROR "${name} unexpectedly succeeded\n${stdout}\n${stderr}")
    endif()
    string(CONCAT output "${stdout}" "${stderr}")
    set("${name}_output" "${output}" PARENT_SCOPE)
endfunction()

set(dataset_dir "${QALSH4C_BINARY_DIR}/disk_qalsh_cli")
file(REMOVE_RECURSE "${dataset_dir}")

run_ok(help --help)
if(NOT help_output MATCHES "[\r\n][ \t]+index[ \t\r\n]")
    message(FATAL_ERROR "Expected top-level help to list index command:\n${help_output}")
endif()

run_ok(index_help index qalsh --help)
foreach(option "--target" "--seed" "--overwrite" "--approximation-ratio")
    if(NOT index_help_output MATCHES "${option}")
        message(FATAL_ERROR "Expected index qalsh help to include ${option}:\n${index_help_output}")
    endif()
endforeach()

run_ok(generate dataset generate gmm
    -a 24
    -b 20
    --num-dimensions 8
    --seed 123
    --no-progress
    -o "${dataset_dir}"
)

run_ok(index_default index qalsh -p 2 -d "${dataset_dir}")
set(l2_b_dir "${dataset_dir}/index/qalsh/l2/b")
foreach(file_name meta.json dot_vectors.bin tables.bin)
    if(NOT EXISTS "${l2_b_dir}/${file_name}")
        message(FATAL_ERROR "Expected ${l2_b_dir}/${file_name} to exist")
    endif()
endforeach()

file(READ "${l2_b_dir}/meta.json" meta)
string(JSON meta_length LENGTH "${meta}")
if(NOT meta_length EQUAL 7)
    message(FATAL_ERROR "Expected exactly 7 disk QALSH metadata fields, got ${meta_length}:\n${meta}")
endif()
string(JSON approximation_ratio GET "${meta}" approximation_ratio)
string(JSON seed GET "${meta}" seed)
string(JSON num_points GET "${meta}" num_points)
string(JSON num_dimensions GET "${meta}" num_dimensions)
string(JSON num_hash_tables GET "${meta}" num_hash_tables)
string(JSON collision_threshold GET "${meta}" collision_threshold)
string(JSON bucket_width GET "${meta}" bucket_width)
if(NOT approximation_ratio MATCHES "^2")
    message(FATAL_ERROR "Expected default approximation ratio 2, got ${approximation_ratio}")
endif()
if(NOT seed EQUAL 42)
    message(FATAL_ERROR "Expected default seed 42, got ${seed}")
endif()
if(NOT num_points EQUAL 20 OR NOT num_dimensions EQUAL 8)
    message(FATAL_ERROR "Unexpected disk QALSH metadata point shape: ${num_points}x${num_dimensions}")
endif()
if(num_hash_tables LESS_EQUAL 0 OR collision_threshold LESS_EQUAL 0 OR NOT bucket_width MATCHES "^[0-9]")
    message(FATAL_ERROR "Unexpected disk QALSH derived metadata:\n${meta}")
endif()

math(EXPR expected_table_size "${num_hash_tables} * 20 * 8")
math(EXPR expected_dot_size "${num_hash_tables} * 8 * 4")
file(SIZE "${l2_b_dir}/tables.bin" actual_table_size)
file(SIZE "${l2_b_dir}/dot_vectors.bin" actual_dot_size)
if(NOT actual_table_size EQUAL expected_table_size)
    message(FATAL_ERROR "Unexpected tables.bin size: expected ${expected_table_size}, got ${actual_table_size}")
endif()
if(NOT actual_dot_size EQUAL expected_dot_size)
    message(FATAL_ERROR "Unexpected dot_vectors.bin size: expected ${expected_dot_size}, got ${actual_dot_size}")
endif()

run_ok(index_existing index qalsh -p 2 -d "${dataset_dir}")
if(NOT index_existing_output MATCHES "already exists")
    message(FATAL_ERROR "Expected existing matching index to be reused:\n${index_existing_output}")
endif()

run_fail(index_mismatch index qalsh -p 2 -d "${dataset_dir}" --seed 7)
if(NOT index_mismatch_output MATCHES "does not match requested parameters")
    message(FATAL_ERROR "Expected mismatched existing index error:\n${index_mismatch_output}")
endif()

run_ok(index_overwrite index qalsh -p 2 -d "${dataset_dir}" --seed 7 --overwrite)
file(READ "${l2_b_dir}/meta.json" meta_after_overwrite)
string(JSON seed_after_overwrite GET "${meta_after_overwrite}" seed)
if(NOT seed_after_overwrite EQUAL 7)
    message(FATAL_ERROR "Expected overwrite to rebuild index with seed 7, got ${seed_after_overwrite}")
endif()

run_ok(index_a_l1 index qalsh -p 1 -d "${dataset_dir}" --target a)
if(NOT EXISTS "${dataset_dir}/index/qalsh/l1/a/meta.json")
    message(FATAL_ERROR "Expected explicit L1 A target index")
endif()

run_fail(missing_l1_b estimate --disk-based -p 1 -d "${dataset_dir}" --direction ab ann qalsh)
if(NOT missing_l1_b_output MATCHES "Run `index qalsh")
    message(FATAL_ERROR "Expected missing index guidance:\n${missing_l1_b_output}")
endif()

run_fail(missing_l2_a_both -T 2 estimate --disk-based -p 2 -d "${dataset_dir}" --direction both ann qalsh)
if(NOT missing_l2_a_both_output MATCHES "index/qalsh/l2/a")
    message(FATAL_ERROR "Expected bidirectional preflight to require missing L2 A index:\n${missing_l2_a_both_output}")
endif()

run_ok(index_both_l2 index qalsh -p 2 -d "${dataset_dir}" --target both --overwrite)
if(NOT EXISTS "${dataset_dir}/index/qalsh/l2/a/meta.json" OR NOT EXISTS "${dataset_dir}/index/qalsh/l2/b/meta.json")
    message(FATAL_ERROR "Expected both L2 target indexes")
endif()

execute_process(COMMAND "${CMAKE_COMMAND}" -E copy "${l2_b_dir}/meta.json" "${l2_b_dir}/meta.good.json")
file(READ "${l2_b_dir}/meta.json" meta_good)
string(JSON meta_bad SET "${meta_good}" num_points 21)
file(WRITE "${l2_b_dir}/meta.json" "${meta_bad}")
run_fail(corrupt_meta -T 2 estimate --disk-based -p 2 -d "${dataset_dir}" --direction ab ann qalsh)
if(NOT corrupt_meta_output MATCHES "metadata mismatch")
    message(FATAL_ERROR "Expected corrupt metadata to fail before search:\n${corrupt_meta_output}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -E copy "${l2_b_dir}/meta.good.json" "${l2_b_dir}/meta.json")
file(REMOVE "${l2_b_dir}/meta.good.json")

execute_process(COMMAND "${CMAKE_COMMAND}" -E copy "${l2_b_dir}/tables.bin" "${l2_b_dir}/tables.good.bin")
file(WRITE "${l2_b_dir}/tables.bin" "")
run_fail(corrupt_tables -T 2 estimate --disk-based -p 2 -d "${dataset_dir}" --direction ab ann qalsh)
if(NOT corrupt_tables_output MATCHES "unexpected size")
    message(FATAL_ERROR "Expected corrupt tables.bin to fail before search:\n${corrupt_tables_output}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -E copy "${l2_b_dir}/tables.good.bin" "${l2_b_dir}/tables.bin")
file(REMOVE "${l2_b_dir}/tables.good.bin")

execute_process(COMMAND "${CMAKE_COMMAND}" -E copy "${l2_b_dir}/dot_vectors.bin" "${l2_b_dir}/dot_vectors.good.bin")
file(WRITE "${l2_b_dir}/dot_vectors.bin" "")
run_fail(corrupt_dot_vectors -T 2 estimate --disk-based -p 2 -d "${dataset_dir}" --direction ab ann qalsh)
if(NOT corrupt_dot_vectors_output MATCHES "unexpected size")
    message(FATAL_ERROR "Expected corrupt dot_vectors.bin to fail before search:\n${corrupt_dot_vectors_output}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -E copy "${l2_b_dir}/dot_vectors.good.bin" "${l2_b_dir}/dot_vectors.bin")
file(REMOVE "${l2_b_dir}/dot_vectors.good.bin")

run_ok(ann_ab -T 2 estimate --disk-based -p 2 -d "${dataset_dir}" --direction ab ann qalsh)
if(NOT ann_ab_output MATCHES "Estimated Distance:")
    message(FATAL_ERROR "Expected disk QALSH ANN A->B estimate:\n${ann_ab_output}")
endif()

run_ok(ann_ba -T 2 estimate --disk-based -p 2 -d "${dataset_dir}" --direction ba ann qalsh)
if(NOT ann_ba_output MATCHES "Direction: B -> A")
    message(FATAL_ERROR "Expected disk QALSH ANN B->A estimate:\n${ann_ba_output}")
endif()

run_ok(ann_both -T 2 estimate --disk-based -p 2 -d "${dataset_dir}" --direction both ann qalsh)
if(NOT ann_both_output MATCHES "Direction: A <-> B")
    message(FATAL_ERROR "Expected disk QALSH ANN bidirectional estimate:\n${ann_both_output}")
endif()

run_fail(ann_c_mismatch -T 2 estimate --disk-based -p 2 -d "${dataset_dir}" --direction ab ann qalsh -c 2.2)
if(NOT ann_c_mismatch_output MATCHES "approximation ratio mismatch")
    message(FATAL_ERROR "Expected explicit approximation ratio mismatch:\n${ann_c_mismatch_output}")
endif()

run_ok(sampling_miss -T 2 estimate --disk-based --disk-query-batch-size 8 -p 2 -d "${dataset_dir}"
    --direction ab sampling --use-cache -n 4 qalsh
)
set(cache_path "${dataset_dir}/l2_qalsh_weights_A.bin")
if(NOT EXISTS "${cache_path}")
    message(FATAL_ERROR "Expected disk QALSH sampling cache at ${cache_path}")
endif()
math(EXPR expected_cache_size "24 * 4")
file(SIZE "${cache_path}" actual_cache_size)
if(NOT actual_cache_size EQUAL expected_cache_size)
    message(FATAL_ERROR "Unexpected QALSH weights cache size: expected ${expected_cache_size}, got ${actual_cache_size}")
endif()

file(RENAME "${dataset_dir}/index" "${dataset_dir}/index.disabled")
run_ok(sampling_hit_no_index -T 2 estimate --disk-based --disk-query-batch-size 8 -p 2 -d "${dataset_dir}"
    --direction ab sampling --use-cache -n 4 qalsh
)
if(sampling_hit_no_index_output MATCHES "Generating weights using QALSH")
    message(FATAL_ERROR "QALSH cache hit unexpectedly required weight regeneration:\n${sampling_hit_no_index_output}")
endif()

run_ok(sampling_hit_no_index_auto -T 2 estimate --disk-based --disk-query-batch-size 8 -p 2 -d "${dataset_dir}"
    --direction ab sampling --use-cache -e 10 --delta 0.5 qalsh
)
if(sampling_hit_no_index_auto_output MATCHES "Generating weights using QALSH")
    message(FATAL_ERROR "Automatic QALSH cache hit unexpectedly required weight regeneration:\n${sampling_hit_no_index_auto_output}")
endif()
if(sampling_hit_no_index_auto_output MATCHES "Run `index qalsh")
    message(FATAL_ERROR "Automatic QALSH cache hit unexpectedly required a disk index:\n${sampling_hit_no_index_auto_output}")
endif()
