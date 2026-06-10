#!/usr/bin/env python3
"""Experiment 2: compare QALSH sampling and QALSH ANN accuracy."""

from __future__ import annotations

import argparse

from experiment_utils import (
    RUN_FIELDS,
    SUMMARY_FIELDS,
    add_common_arguments,
    build_ann_command,
    build_sampling_command,
    common_run_row,
    configure_process_limits,
    csv_writer,
    ensure_success,
    finalize_common_args,
    make_cases,
    maybe_build_release,
    run_or_print,
    summary_from_rows,
    validate_cases,
    warm_qalsh_weights,
    write_rows,
)


EXPERIMENT = "experiment2_qalsh_sampling_vs_ann"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare QALSH sampling and QALSH ANN relative error.")
    add_common_arguments(parser)
    parser.add_argument("--sampling-runs", type=int, default=20, help="QALSH sampling runs per dataset/norm.")
    parser.add_argument("--delta", default="0.1", help="Failure probability for automatic sample count.")
    parser.add_argument("--epsilon", default="1.0", help="Epsilon for automatic QALSH sampling sample count.")
    args = parser.parse_args()
    finalize_common_args(args, EXPERIMENT)
    if args.sampling_runs < 1:
        raise SystemExit("--sampling-runs must be at least 1")
    return args


def main() -> int:
    args = parse_args()
    cases = make_cases(args)
    validate_cases(args, cases)
    configure_process_limits(args)
    maybe_build_release(args)

    run_writer, run_file = csv_writer(args.run_output, args.append, RUN_FIELDS)
    summary_writer, summary_file = csv_writer(args.summary_output, args.append, SUMMARY_FIELDS)
    has_failure = False

    with run_file, summary_file:
        for case in cases:
            if not warm_qalsh_weights(args, case):
                has_failure = True
                continue

            sampling_rows: list[dict[str, str]] = []
            for run_number in range(1, args.sampling_runs + 1):
                command = build_sampling_command(
                    args,
                    case,
                    "qalsh",
                    epsilon=args.epsilon,
                    use_cache=True,
                )
                result = run_or_print(args, command, f"{case.dataset} p={case.norm_order} qalsh_sampling run={run_number}")
                if not ensure_success(result, f"{case.dataset} p={case.norm_order} qalsh_sampling"):
                    has_failure = True
                sampling_rows.append(
                    common_run_row(
                        args=args,
                        experiment=EXPERIMENT,
                        case=case,
                        algorithm="qalsh_sampling",
                        run_number=run_number,
                        command=command,
                        result=result,
                        parameter_name="method",
                        parameter_value="qalsh_sampling",
                        epsilon=args.epsilon,
                        delta=str(args.delta),
                    )
                )

            write_rows(run_writer, run_file, sampling_rows)
            write_rows(
                summary_writer,
                summary_file,
                [
                    summary_from_rows(
                        args=args,
                        experiment=EXPERIMENT,
                        case=case,
                        algorithm="qalsh_sampling",
                        rows=sampling_rows,
                        runs_requested=args.sampling_runs,
                        run_output=args.run_output,
                        parameter_name="method",
                        parameter_value="qalsh_sampling",
                        epsilon=args.epsilon,
                        delta=str(args.delta),
                    )
                ],
            )

            ann_command = build_ann_command(args, case, "qalsh", use_fixed_seed=args.fixed_ann_seed)
            ann_result = run_or_print(args, ann_command, f"{case.dataset} p={case.norm_order} qalsh_ann")
            if not ensure_success(ann_result, f"{case.dataset} p={case.norm_order} qalsh_ann"):
                has_failure = True
            ann_rows = [
                common_run_row(
                    args=args,
                    experiment=EXPERIMENT,
                    case=case,
                    algorithm="qalsh_ann",
                    run_number=1,
                    command=ann_command,
                    result=ann_result,
                    parameter_name="method",
                    parameter_value="qalsh_ann",
                )
            ]
            write_rows(run_writer, run_file, ann_rows)
            write_rows(
                summary_writer,
                summary_file,
                [
                    summary_from_rows(
                        args=args,
                        experiment=EXPERIMENT,
                        case=case,
                        algorithm="qalsh_ann",
                        rows=ann_rows,
                        runs_requested=1,
                        run_output=args.run_output,
                        parameter_name="method",
                        parameter_value="qalsh_ann",
                    )
                ],
            )

    print(f"[output] runs: {args.run_output}", flush=True)
    print(f"[output] summary: {args.summary_output}", flush=True)
    return 1 if has_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
