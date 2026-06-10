#!/usr/bin/env python3
"""Experiment 3: compare Bakshi4C, uniform sampling, and QALSH sampling over T."""

from __future__ import annotations

import argparse
from collections import defaultdict

from experiment_utils import (
    RUN_FIELDS,
    SUMMARY_FIELDS,
    add_bakshi_arguments,
    add_common_arguments,
    bakshi_run_row_to_common,
    build_bakshi_command,
    build_sampling_command,
    common_run_row,
    configure_process_limits,
    csv_writer,
    ensure_success,
    finalize_bakshi_args,
    finalize_common_args,
    int_range,
    make_cases,
    maybe_build_release,
    parse_int_list,
    read_csv_rows,
    run_or_print,
    summary_from_rows,
    timestamp_for_path,
    validate_cases,
    warm_qalsh_weights,
    write_rows,
)


EXPERIMENT = "experiment3_sampling_accuracy_by_t"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Bakshi4C, uniform sampling, and QALSH sampling accuracy for sample counts T."
    )
    add_common_arguments(parser)
    add_bakshi_arguments(parser)
    parser.add_argument("--runs", type=int, default=20, help="Sampling runs per dataset/norm/T.")
    parser.add_argument("--sample-counts", nargs="+", default=None, help="Explicit T values.")
    parser.add_argument("--sample-start", type=int, default=10, help="First T when --sample-counts is omitted.")
    parser.add_argument("--sample-stop", type=int, default=100, help="Last T when --sample-counts is omitted.")
    parser.add_argument("--sample-step", type=int, default=10, help="Step when --sample-counts is omitted.")
    parser.add_argument("--delta", default="0.1", help="Failure probability used by qalsh4c when needed.")
    parser.add_argument("--epsilon", default="1.0", help="Epsilon used by qalsh4c when needed.")
    args = parser.parse_args()
    finalize_common_args(args, EXPERIMENT)
    finalize_bakshi_args(args)
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")
    if args.sample_counts is None:
        args.sample_counts = int_range(args.sample_start, args.sample_stop, args.sample_step, "--sample")
    else:
        args.sample_counts = parse_int_list(args.sample_counts, "--sample-counts")
    return args


def run_qalsh4c_sampling_method(
    args: argparse.Namespace,
    case,
    algorithm: str,
    sample_count: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run_number in range(1, args.runs + 1):
        command = build_sampling_command(
            args,
            case,
            algorithm,
            num_samples=sample_count,
            epsilon=args.epsilon,
            use_cache=algorithm == "qalsh",
        )
        label = f"{case.dataset} p={case.norm_order} {algorithm} T={sample_count} run={run_number}"
        result = run_or_print(args, command, label)
        ensure_success(result, label)
        rows.append(
            common_run_row(
                args=args,
                experiment=EXPERIMENT,
                case=case,
                algorithm="qalsh_sampling" if algorithm == "qalsh" else "uniform_sampling",
                run_number=run_number,
                command=command,
                result=result,
                parameter_name="sample_count",
                parameter_value=str(sample_count),
                epsilon=args.epsilon if algorithm == "qalsh" else "",
                delta=str(args.delta) if algorithm == "qalsh" else "",
                sample_count_requested=str(sample_count),
            )
        )
    return rows


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

            if not args.skip_bakshi:
                stamp = timestamp_for_path()
                bakshi_run_output = args.output_dir / f"{EXPERIMENT}.bakshi.{case.dataset}.p{case.norm_order}.{stamp}.runs.csv"
                bakshi_summary_output = (
                    args.output_dir / f"{EXPERIMENT}.bakshi.{case.dataset}.p{case.norm_order}.{stamp}.summary.csv"
                )
                command = build_bakshi_command(
                    args,
                    case,
                    args.sample_counts,
                    bakshi_run_output,
                    bakshi_summary_output,
                    args.runs,
                )
                result = run_or_print(args, command, f"{case.dataset} p={case.norm_order} bakshi4c")
                if not ensure_success(result, f"{case.dataset} p={case.norm_order} bakshi4c"):
                    has_failure = True
                elif not args.dry_run:
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
                    rows_by_t: dict[str, list[dict[str, str]]] = defaultdict(list)
                    for row in bakshi_rows:
                        rows_by_t[row["sample_count_requested"]].append(row)
                    for sample_count, rows in sorted(rows_by_t.items(), key=lambda item: int(item[0])):
                        write_rows(
                            summary_writer,
                            summary_file,
                            [
                                summary_from_rows(
                                    args=args,
                                    experiment=EXPERIMENT,
                                    case=case,
                                    algorithm="bakshi4c",
                                    rows=rows,
                                    runs_requested=args.runs,
                                    run_output=args.run_output,
                                    parameter_name="sample_count",
                                    parameter_value=sample_count,
                                    sample_count_requested=sample_count,
                                    time_source="bakshi_wall_clock",
                                )
                            ],
                        )

            for sample_count in args.sample_counts:
                for qalsh4c_algorithm, summary_algorithm in (
                    ("uniform", "uniform_sampling"),
                    ("qalsh", "qalsh_sampling"),
                ):
                    rows = run_qalsh4c_sampling_method(args, case, qalsh4c_algorithm, sample_count)
                    if any(row["return_code"] != "0" for row in rows):
                        has_failure = True
                    write_rows(run_writer, run_file, rows)
                    write_rows(
                        summary_writer,
                        summary_file,
                        [
                            summary_from_rows(
                                args=args,
                                experiment=EXPERIMENT,
                                case=case,
                                algorithm=summary_algorithm,
                                rows=rows,
                                runs_requested=args.runs,
                                run_output=args.run_output,
                                parameter_name="sample_count",
                                parameter_value=str(sample_count),
                                epsilon=args.epsilon if qalsh4c_algorithm == "qalsh" else "",
                                delta=str(args.delta) if qalsh4c_algorithm == "qalsh" else "",
                                sample_count_requested=str(sample_count),
                            )
                        ],
                    )

    print(f"[output] runs: {args.run_output}", flush=True)
    print(f"[output] summary: {args.summary_output}", flush=True)
    return 1 if has_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
