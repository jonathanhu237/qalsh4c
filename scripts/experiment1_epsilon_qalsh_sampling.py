#!/usr/bin/env python3
"""Experiment 1: sweep epsilon and summarize QALSH sampling relative error."""

from __future__ import annotations

import argparse

from experiment_utils import (
    RUN_FIELDS,
    SUMMARY_FIELDS,
    add_common_arguments,
    build_sampling_command,
    configure_process_limits,
    csv_writer,
    decimal_range,
    ensure_success,
    finalize_common_args,
    make_cases,
    maybe_build_release,
    run_or_print,
    summary_from_rows,
    validate_cases,
    warm_qalsh_weights,
    write_rows,
    common_run_row,
)


EXPERIMENT = "experiment1_epsilon_qalsh_sampling"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep epsilon and compare mean relative error of QALSH sampling across datasets."
    )
    add_common_arguments(parser)
    parser.add_argument("--runs", type=int, default=20, help="Sampling runs per dataset/norm/epsilon.")
    parser.add_argument(
        "--epsilons",
        nargs="+",
        default=None,
        help="Explicit epsilon values. Supports spaces or comma-separated values.",
    )
    parser.add_argument("--epsilon-start", default="0.3", help="First epsilon when --epsilons is omitted.")
    parser.add_argument("--epsilon-stop", default="1.0", help="Last epsilon when --epsilons is omitted.")
    parser.add_argument("--epsilon-step", default="0.1", help="Step when --epsilons is omitted.")
    parser.add_argument("--delta", default="0.1", help="Failure probability for automatic sample count.")
    args = parser.parse_args()
    finalize_common_args(args, EXPERIMENT)
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")
    if args.epsilons is None:
        args.epsilons = decimal_range(args.epsilon_start, args.epsilon_stop, args.epsilon_step)
    else:
        from experiment_utils import parse_list

        args.epsilons = parse_list(args.epsilons, None, "--epsilons")
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

            for epsilon in args.epsilons:
                rows: list[dict[str, str]] = []
                for run_number in range(1, args.runs + 1):
                    command = build_sampling_command(
                        args,
                        case,
                        "qalsh",
                        epsilon=epsilon,
                        use_cache=True,
                    )
                    result = run_or_print(
                        args,
                        command,
                        f"{case.dataset} p={case.norm_order} eps={epsilon} run={run_number}",
                    )
                    if not ensure_success(result, f"{case.dataset} p={case.norm_order} eps={epsilon}"):
                        has_failure = True
                    rows.append(
                        common_run_row(
                            args=args,
                            experiment=EXPERIMENT,
                            case=case,
                            algorithm="qalsh_sampling",
                            run_number=run_number,
                            command=command,
                            result=result,
                            parameter_name="epsilon",
                            parameter_value=epsilon,
                            epsilon=epsilon,
                            delta=str(args.delta),
                        )
                    )

                write_rows(run_writer, run_file, rows)
                summary = summary_from_rows(
                    args=args,
                    experiment=EXPERIMENT,
                    case=case,
                    algorithm="qalsh_sampling",
                    rows=rows,
                    runs_requested=args.runs,
                    run_output=args.run_output,
                    parameter_name="epsilon",
                    parameter_value=epsilon,
                    epsilon=epsilon,
                    delta=str(args.delta),
                )
                write_rows(summary_writer, summary_file, [summary])

    print(f"[output] runs: {args.run_output}", flush=True)
    print(f"[output] summary: {args.summary_output}", flush=True)
    return 1 if has_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
