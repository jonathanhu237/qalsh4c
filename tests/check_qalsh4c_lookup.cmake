# Looking up saved exact distances gives the same estimate as running the exact queries, the kappa command reports the
# certified constants of the paper, and precomputed weights and repeats work.
set(source_data_dir "${QALSH4C_SOURCE_DIR}/data/federalist")
set(work_dir "${QALSH4C_BINARY_DIR}/lookup_check")
file(REMOVE_RECURSE "${work_dir}")
file(MAKE_DIRECTORY "${work_dir}")
foreach(name A.bin B.bin metadata.json)
    file(COPY_FILE "${source_data_dir}/${name}" "${work_dir}/${name}")
endforeach()

function(run name output_var)
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
    set(${output_var} "${output}" PARENT_SCOPE)
endfunction()

function(estimate_of output_var output)
    string(REGEX MATCH "Sample estimate: repeat=1 T=[0-9]+ estimate=([-+0-9.eE]+)" match "${output}")
    if(NOT match)
        message(FATAL_ERROR "No sample estimate in output:\n${output}")
    endif()
    set(${output_var} "${CMAKE_MATCH_1}" PARENT_SCOPE)
endfunction()

# Certified constants (Table 2 of the paper, c = 2, continuous front with the default gamma = c^1.5).
run(kappa_l1 kappa_l1_output kappa -p 1 -c 2 -e 0.3 0.1)
if(NOT kappa_l1_output MATCHES "kappa_up=1\\.982358" OR NOT kappa_l1_output MATCHES "T epsilon=0\\.3 delta=0\\.1 T=110"
   OR NOT kappa_l1_output MATCHES "T epsilon=0\\.1 delta=0\\.1 T=983")
    message(FATAL_ERROR "Unexpected l1 kappa output:\n${kappa_l1_output}")
endif()
run(kappa_l2 kappa_l2_output kappa -p 2 -c 2 -e 0.3 0.1)
if(NOT kappa_l2_output MATCHES "T epsilon=0\\.3 delta=0\\.1 T=119" OR NOT kappa_l2_output MATCHES "T epsilon=0\\.1 delta=0\\.1 T=1065")
    message(FATAL_ERROR "Unexpected l2 kappa output:\n${kappa_l2_output}")
endif()
# gamma = c gives the constants of the earlier default.
run(kappa_c kappa_c_output --qalsh-continuous-ratio 2 kappa -p 1 -c 2 -e 0.3 0.1)
if(NOT kappa_c_output MATCHES "kappa_up=1\\.416132" OR NOT kappa_c_output MATCHES "T epsilon=0\\.3 delta=0\\.1 T=47"
   OR NOT kappa_c_output MATCHES "T epsilon=0\\.1 delta=0\\.1 T=417")
    message(FATAL_ERROR "Unexpected kappa output at gamma = c:\n${kappa_c_output}")
endif()

# Exact distances from the exact scan.
set(exact_file "${work_dir}/exact_l2.bin")
run(save save_output estimate -p 2 -d "${work_dir}" --direction ab ann --save-distances "${exact_file}" linear_scan)
file(SIZE "${exact_file}" exact_size)
math(EXPR expected_size "1880 * 8")
if(NOT exact_size EQUAL expected_size)
    message(FATAL_ERROR "Exact distances file has ${exact_size} bytes, expected ${expected_size}")
endif()

# The same seed draws the same sample from the same cached weights; computed and looked-up distances must agree.
run(computed computed_output --use-fixed-seed estimate -p 2 -d "${work_dir}" --direction ab
    sampling -n 64 --use-cache qalsh -c 2)
run(looked_up looked_up_output --use-fixed-seed estimate -p 2 -d "${work_dir}" --direction ab
    sampling -n 64 --use-cache --exact-distances "${exact_file}" qalsh -c 2)
if(looked_up_output MATCHES "Generating weights using QALSH")
    message(FATAL_ERROR "The lookup run did not reuse the cached weights:\n${looked_up_output}")
endif()
estimate_of(computed_value "${computed_output}")
estimate_of(looked_up_value "${looked_up_output}")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env A=${computed_value} B=${looked_up_value}
            awk "BEGIN { d = ENVIRON[\"A\"] - ENVIRON[\"B\"]; if (d < 0) d = -d; exit !(d <= 1e-5 * ENVIRON[\"A\"]) }"
    RESULT_VARIABLE compare_result
)
if(NOT compare_result EQUAL 0)
    message(FATAL_ERROR "Computed estimate ${computed_value} differs from looked-up estimate ${looked_up_value}")
endif()

# Precomputed weights (the cached crude distances) with repeats: one estimate per repeat.
run(precomputed precomputed_output estimate -p 2 -d "${work_dir}" --direction ab
    sampling -n 32 --repeats 5 --exact-distances "${exact_file}" precomputed -f "${work_dir}/l2_qalsh_weights_A.bin")
string(REGEX MATCHALL "Sample estimate: repeat=[0-9]+ T=32 estimate=" repeats "${precomputed_output}")
list(LENGTH repeats repeat_count)
if(NOT repeat_count EQUAL 5)
    message(FATAL_ERROR "Expected 5 estimates, got ${repeat_count}:\n${precomputed_output}")
endif()
