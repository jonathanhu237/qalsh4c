#!/usr/bin/env python3
"""Experiment 4: compare speed and accuracy of linear scan, Bakshi4C, and QALSH sampling."""

from __future__ import annotations

import argparse

from experiment_utils import (
    RUN_FIELDS,
    SUMMARY_FIELDS,
    add_bakshi_arguments,
    add_common_arguments,
    bakshi_run_row_to_common,
    build_ann_command,
    build_bakshi_command,
    build_sampling_command,
    common_run_row,
    configure_process_limits,
    csv_writer,
    ensure_success,
    finalize_bakshi_args,
    finalize_common_args,
    make_cases,
    maybe_build_release,
    parse_qalsh_output,
    read_csv_rows,
    run_or_print,
    summary_from_rows,
    timestamp_for_path,
    validate_cases,
    warm_qalsh_weights,
    write_rows,
)


EXPERIMENT = "experiment4_speed_accuracy_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare linear scan, Bakshi4C, and QALSH sampling speed/accuracy. "
            "Bakshi4C uses the automatic QALSH sampling sample count for each dataset/norm."
        )
    )
    add_common_arguments(parser)
    add_bakshi_arguments(parser)
    parser.add_argument("--runs", type=int, default=20, help="Sampling runs for Bakshi4C and QALSH sampling.")
    parser.add_argument("--delta", default="0.1", help="Failure probability for automatic QALSH sample count.")
    parser.add_argument("--epsilon", default="1.0", help="Epsilon for automatic QALSH sample count.")
    args = parser.parse_args()
    finalize_common_args(args, EXPERIMENT)
    finalize_bakshi_args(args)
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")
    return args


def determine_qalsh_auto_sample_count(args: argparse.Namespace, case) -> int | None:
    command = build_sampling_command(
        args,
        case,
        "qalsh",
        epsilon=args.epsilon,
        use_cache=True,
    )
    result = run_or_print(args, command, f"{case.dataset} p={case.norm_order} qalsh_sampling auto-T probe")
    if not ensure_success(result, f"{case.dataset} p={case.norm_order} qalsh_sampling auto-T probe"):
        return None
    if args.dry_run:
        return 1
    parsed = parse_qalsh_output(result.output)
    selected_samples = parsed["selected_samples"]
    if not selected_samples:
        print(f"[probe] could not parse selected_samples for {case.dataset} p={case.norm_order}", flush=True)
        return None
    return int(selected_samples)


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
            linear_command = build_ann_command(args, case, "linear_scan")
            linear_result = run_or_print(args, linear_command, f"{case.dataset} p={case.norm_order} linear_scan")
            if not ensure_success(linear_result, f"{case.dataset} p={case.norm_order} linear_scan"):
                has_failure = True
            linear_rows = [
                common_run_row(
                    args=args,
                    experiment=EXPERIMENT,
                    case=case,
                    algorithm="linear_scan",
                    run_number=1,
                    command=linear_command,
                    result=linear_result,
                    parameter_name="method",
                    parameter_value="linear_scan",
                )
            ]
            write_rows(run_writer, run_file, linear_rows)
            write_rows(
                summary_writer,
                summary_file,
                [
                    summary_from_rows(
                        args=args,
                        experiment=EXPERIMENT,
                        case=case,
                        algorithm="linear_scan",
                        rows=linear_rows,
                        runs_requested=1,
                        run_output=args.run_output,
                        parameter_name="method",
                        parameter_value="linear_scan",
                    )
                ],
            )

            if not warm_qalsh_weights(args, case):
                has_failure = True
                continue

            selected_samples = determine_qalsh_auto_sample_count(args, case)
            if selected_samples is None:
                has_failure = True
                continue

            qalsh_rows: list[dict[str, str]] = []
            for run_number in range(1, args.runs + 1):
                command = build_sampling_command(
                    args,
                    case,
                    "qalsh",
                    num_samples=selected_samples,
                    epsilon=args.epsilon,
                    use_cache=True,
                )
                label = f"{case.dataset} p={case.norm_order} qalsh_sampling T={selected_samples} run={run_number}"
                result = run_or_print(args, command, label)
                if not ensure_success(result, label):
                    has_failure = True
                qalsh_rows.append(
                    common_run_row(
                        args=args,
                        experiment=EXPERIMENT,
                        case=case,
                        algorithm="qalsh_sampling",
                        run_number=run_number,
                        command=command,
                        result=result,
                        parameter_name="sample_count",
                        parameter_value=str(selected_samples),
                        epsilon=args.epsilon,
                        delta=str(args.delta),
                        sample_count_requested=str(selected_samples),
                    )
                )

            write_rows(run_writer, run_file, qalsh_rows)
            write_rows(
                summary_writer,
                summary_file,
                [
                    summary_from_rows(
                        args=args,
                        experiment=EXPERIMENT,
                        case=case,
                        algorithm="qalsh_sampling",
                        rows=qalsh_rows,
                        runs_requested=args.runs,
                        run_output=args.run_output,
                        parameter_name="sample_count",
                        parameter_value=str(selected_samples),
                        epsilon=args.epsilon,
                        delta=str(args.delta),
                        sample_count_requested=str(selected_samples),
                    )
                ],
            )

            if args.skip_bakshi:
                continue

            stamp = timestamp_for_path()
            bakshi_run_output = args.output_dir / f"{EXPERIMENT}.bakshi.{case.dataset}.p{case.norm_order}.{stamp}.runs.csv"
            bakshi_summary_output = (
                args.output_dir / f"{EXPERIMENT}.bakshi.{case.dataset}.p{case.norm_order}.{stamp}.summary.csv"
            )
            command = build_bakshi_command(
                args,
                case,
                [selected_samples],
                bakshi_run_output,
                bakshi_summary_output,
                args.runs,
            )
            result = run_or_print(args, command, f"{case.dataset} p={case.norm_order} bakshi4c T={selected_samples}")
            if not ensure_success(result, f"{case.dataset} p={case.norm_order} bakshi4c T={selected_samples}"):
                has_failure = True
                continue
            if args.dry_run:
                continue

            bakshi_rows = [
                bakshi_run_row_to_common(
                    args=args,
                    experiment=EXPERIMENT,
                    case=case,
                    source_row=row,
                    source_run_csv=bakshi_run_output,
                )
                for row in read_csv_rows(bakshi_run_output)
            ]
            write_rows(run_writer, run_file, bakshi_rows)
            write_rows(
                summary_writer,
                summary_file,
                [
                    summary_from_rows(
                        args=args,
                        experiment=EXPERIMENT,
                        case=case,
                        algorithm="bakshi4c",
                        rows=bakshi_rows,
                        runs_requested=args.runs,
                        run_output=args.run_output,
                        parameter_name="sample_count",
                        parameter_value=str(selected_samples),
                        sample_count_requested=str(selected_samples),
                        time_source="bakshi_wall_clock",
                    )
                ],
            )

    print(f"[output] runs: {args.run_output}", flush=True)
    print(f"[output] summary: {args.summary_output}", flush=True)
    return 1 if has_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
