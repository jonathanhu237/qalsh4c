#!/usr/bin/env python3
"""Experiment 4: sweep QALSH sampling epsilon using existing weight caches."""

from __future__ import annotations

import argparse
import statistics
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import experiment_utils as _experiment_utils

# Current qalsh4c datasets store Coordinate=float and Weight=float.
_experiment_utils.BYTES_PER_COORDINATE = 4

from experiment_utils import (
    DATASET_LABELS,
    RUN_FIELDS,
    SUMMARY_FIELDS,
    build_sampling_command,
    common_run_row,
    configure_process_limits,
    csv_writer,
    decimal_range,
    ensure_success,
    finalize_common_args,
    make_cases,
    maybe_build_release,
    parse_list,
    point_set_sources_for_direction,
    read_csv_rows,
    read_metadata,
    run_or_print,
    summary_from_rows,
    to_float,
    validate_cases,
    weight_cache_paths,
    write_rows,
)


EXPERIMENT = "experiment4_qalsh_epsilon_sweep_cached"
DEFAULT_BARK_BASE_URL = ""
DEFAULT_DATASETS = ["federalist", "mnist", "trevi", "gist"]
DEFAULT_WEIGHT_SUMMARY_GLOB = "experiment4_qalsh_default_vs_quadtree_t100*.summary.csv"
BYTES_PER_WEIGHT = 4

RUN_FIELDS_LOCAL = RUN_FIELDS + ["measurement_role"]
SUMMARY_FIELDS_LOCAL = SUMMARY_FIELDS + [
    "weight_generation_time_ms",
    "weight_generation_wall_ms",
    "sampling_mean_time_ms",
    "sampling_stdev_time_ms",
    "sampling_mean_wall_ms",
    "sampling_stdev_wall_ms",
    "speed_total_time_ms",
    "speed_total_wall_ms",
    "weight_summary_csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep QALSH sampling epsilon from 0.3 to 1.0 using existing weight caches. "
            "The script does not generate or refresh caches; total time is computed as "
            "known weight-generation time plus cached sampling time."
        )
    )
    _experiment_utils.add_common_arguments(parser)
    parser.set_defaults(datasets=DEFAULT_DATASETS)
    parser.add_argument("--runs", type=int, default=20, help="Cached sampling runs per dataset/norm/epsilon.")
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
    parser.add_argument(
        "--weight-summary",
        type=Path,
        default=None,
        help=(
            "Summary CSV containing qalsh_sampling weight_generation_time_ms. "
            "Defaults to the newest previous experiment 4 summary in --output-dir."
        ),
    )
    parser.add_argument(
        "--weight-summary-glob",
        default=DEFAULT_WEIGHT_SUMMARY_GLOB,
        help="Glob used in --output-dir when --weight-summary is omitted.",
    )
    parser.add_argument(
        "--bark-url",
        default=DEFAULT_BARK_BASE_URL,
        help="Bark base URL for completion notices; disabled when empty.",
    )
    parser.add_argument("--no-bark", action="store_true", help="Disable Bark completion notices.")
    parser.add_argument("--bark-timeout", type=float, default=10.0, help="Bark request timeout in seconds.")
    args = parser.parse_args()
    finalize_common_args(args, EXPERIMENT)
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")
    if args.epsilons is None:
        args.epsilons = decimal_range(args.epsilon_start, args.epsilon_stop, args.epsilon_step)
    else:
        args.epsilons = parse_list(args.epsilons, None, "--epsilons")
    if args.refresh_weights:
        raise SystemExit("This cached epsilon sweep does not support --refresh-weights")
    if args.bark_timeout <= 0.0:
        raise SystemExit("--bark-timeout must be positive")
    if args.weight_summary is not None:
        args.weight_summary = _experiment_utils.normalize_path(args.weight_summary)
    return args


def normalize_bark_base_url(raw_url: str) -> str:
    base_url = raw_url.strip()
    for placeholder in ("/Body Text", "/Body%20Text"):
        if base_url.endswith(placeholder):
            base_url = base_url[: -len(placeholder)]
            break
    return base_url.rstrip("/")


def send_bark_notification(args: argparse.Namespace, exit_code: int) -> None:
    if args.no_bark:
        return
    base_url = normalize_bark_base_url(args.bark_url)
    if not base_url:
        return

    status = "succeeded" if exit_code == 0 else "finished with failures"
    title = urllib.parse.quote(f"QALSH4C Exp4 epsilon sweep {status}", safe="")
    body = urllib.parse.quote(
        "\n".join(
            [
                f"Experiment: {EXPERIMENT}",
                f"Exit code: {exit_code}",
                f"Runs CSV: {args.run_output}",
                f"Summary CSV: {args.summary_output}",
            ]
        ),
        safe="",
    )
    url = f"{base_url}/{title}/{body}"
    if args.dry_run:
        print(f"[bark] dry-run would notify: {url}", flush=True)
        return

    try:
        with urllib.request.urlopen(url, timeout=args.bark_timeout) as response:
            print(f"[bark] notified status={response.status}", flush=True)
    except (OSError, TimeoutError, urllib.error.URLError) as error:
        print(f"[bark] notification failed: {error}", flush=True)


def norm_key(norm_order: str) -> str:
    return f"{float(norm_order):g}"


def format_ms(value: float | None) -> str:
    return f"{value:.12g}" if value is not None else ""


def mean_value(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def stdev_value(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def find_default_weight_summary(args: argparse.Namespace) -> Path:
    candidates = sorted(
        args.output_dir.glob(args.weight_summary_glob),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit(
            f"Cannot find previous weight summary in {args.output_dir} with glob {args.weight_summary_glob!r}; "
            "pass --weight-summary explicitly."
        )
    return candidates[0]


def resolve_weight_summary(args: argparse.Namespace) -> Path:
    weight_summary = args.weight_summary if args.weight_summary is not None else find_default_weight_summary(args)
    if not weight_summary.is_file():
        raise SystemExit(f"missing weight summary CSV: {weight_summary}")
    print(f"[weights] using generation times from {weight_summary}", flush=True)
    return weight_summary


def weight_time_key(
    dataset_label: str,
    norm_order: str,
    direction: str,
    approximation_ratio: str,
) -> tuple[str, str, str, str]:
    return (dataset_label, norm_key(norm_order), direction, str(approximation_ratio))


def load_weight_generation_times(
    weight_summary: Path,
) -> dict[tuple[str, str, str, str], tuple[float | None, float | None]]:
    times: dict[tuple[str, str, str, str], tuple[float | None, float | None]] = {}
    for row in read_csv_rows(weight_summary):
        if row.get("algorithm") != "qalsh_sampling":
            continue
        time_ms = to_float(row.get("weight_generation_time_ms", ""))
        wall_ms = to_float(row.get("weight_generation_wall_ms", ""))
        if time_ms is None and wall_ms is None:
            continue
        key = weight_time_key(
            row["dataset"],
            row["norm_order"],
            row["direction"],
            row.get("approximation_ratio", ""),
        )
        times[key] = (time_ms, wall_ms)
    if not times:
        raise SystemExit(f"no qalsh_sampling weight-generation times found in {weight_summary}")
    return times


def generation_times_for_case(
    args: argparse.Namespace,
    case,
    weight_times: dict[tuple[str, str, str, str], tuple[float | None, float | None]],
) -> tuple[float | None, float | None]:
    key = weight_time_key(DATASET_LABELS[case.dataset], case.norm_order, args.direction, str(args.approximation_ratio))
    if key not in weight_times:
        raise SystemExit(
            "missing weight-generation time for "
            f"dataset={DATASET_LABELS[case.dataset]} norm={case.norm_order} direction={args.direction} "
            f"approximation_ratio={args.approximation_ratio}"
        )
    return weight_times[key]


def expected_weight_cache_bytes(case, source: str) -> int:
    metadata = read_metadata(case.dataset_dir)
    point_key = source.lower()
    return int(metadata["num_points"][point_key]) * BYTES_PER_WEIGHT


def validate_qalsh_weight_caches(args: argparse.Namespace, cases) -> None:
    problems: list[str] = []
    sources = point_set_sources_for_direction(args.direction)
    for case in cases:
        for source, cache_path in zip(sources, weight_cache_paths(case, args.direction), strict=True):
            expected_bytes = expected_weight_cache_bytes(case, source)
            if not cache_path.is_file():
                problems.append(f"missing cache: {cache_path}")
                continue
            actual_bytes = cache_path.stat().st_size
            if actual_bytes != expected_bytes:
                problems.append(f"bad cache size: {cache_path} has {actual_bytes} bytes, expected {expected_bytes}")

    if problems:
        joined = "\n  - ".join(problems)
        raise SystemExit(f"Refusing to run because QALSH caches are not ready:\n  - {joined}")


def add_local_fields(row: dict[str, str], measurement_role: str) -> dict[str, str]:
    row["measurement_role"] = measurement_role
    return row


def run_cached_sampling_rows(args: argparse.Namespace, case, epsilon: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run_number in range(1, args.runs + 1):
        command = build_sampling_command(
            args,
            case,
            "qalsh",
            epsilon=epsilon,
            use_cache=True,
        )
        label = f"{case.dataset} p={case.norm_order} qalsh eps={epsilon} run={run_number}"
        result = run_or_print(args, command, label)
        ensure_success(result, label)
        rows.append(
            add_local_fields(
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
                    sample_count_requested="auto",
                ),
                "cached_sampling",
            )
        )
    return rows


def make_summary(
    *,
    args: argparse.Namespace,
    case,
    epsilon: str,
    rows: list[dict[str, str]],
    generation_time_ms: float | None,
    generation_wall_ms: float | None,
    weight_summary: Path,
) -> dict[str, str]:
    sampling_time_values = [value for row in rows if (value := to_float(row["time_ms"])) is not None]
    sampling_wall_values = [value for row in rows if (value := to_float(row["wall_ms"])) is not None]
    sampling_mean_time = mean_value(sampling_time_values)
    sampling_mean_wall = mean_value(sampling_wall_values)
    speed_total_time = None
    speed_total_wall = None
    if generation_time_ms is not None and sampling_mean_time is not None:
        speed_total_time = generation_time_ms + sampling_mean_time
    if generation_wall_ms is not None and sampling_mean_wall is not None:
        speed_total_wall = generation_wall_ms + sampling_mean_wall

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
        sample_count_requested="auto",
        time_source="known_weight_generation_plus_cached_sampling_mean",
    )
    summary["weight_generation_time_ms"] = format_ms(generation_time_ms)
    summary["weight_generation_wall_ms"] = format_ms(generation_wall_ms)
    summary["sampling_mean_time_ms"] = format_ms(sampling_mean_time)
    summary["sampling_stdev_time_ms"] = format_ms(stdev_value(sampling_time_values))
    summary["sampling_mean_wall_ms"] = format_ms(sampling_mean_wall)
    summary["sampling_stdev_wall_ms"] = format_ms(stdev_value(sampling_wall_values))
    summary["speed_total_time_ms"] = format_ms(speed_total_time)
    summary["speed_total_wall_ms"] = format_ms(speed_total_wall)
    summary["weight_summary_csv"] = str(weight_summary)
    summary["mean_time_ms"] = summary["speed_total_time_ms"]
    summary["stdev_time_ms"] = summary["sampling_stdev_time_ms"]
    summary["mean_wall_ms"] = summary["speed_total_wall_ms"]
    return summary


def main() -> int:
    args = parse_args()
    cases = make_cases(args)
    validate_cases(args, cases)
    configure_process_limits(args)
    maybe_build_release(args)
    weight_summary = resolve_weight_summary(args)
    weight_times = load_weight_generation_times(weight_summary)
    if not args.dry_run:
        validate_qalsh_weight_caches(args, cases)

    run_writer, run_file = csv_writer(args.run_output, args.append, RUN_FIELDS_LOCAL)
    summary_writer, summary_file = csv_writer(args.summary_output, args.append, SUMMARY_FIELDS_LOCAL)
    has_failure = False

    with run_file, summary_file:
        for case in cases:
            generation_time_ms, generation_wall_ms = generation_times_for_case(args, case, weight_times)
            for epsilon in args.epsilons:
                rows = run_cached_sampling_rows(args, case, epsilon)
                if any(row["return_code"] != "0" for row in rows):
                    has_failure = True
                write_rows(run_writer, run_file, rows)
                write_rows(
                    summary_writer,
                    summary_file,
                    [
                        make_summary(
                            args=args,
                            case=case,
                            epsilon=epsilon,
                            rows=rows,
                            generation_time_ms=generation_time_ms,
                            generation_wall_ms=generation_wall_ms,
                            weight_summary=weight_summary,
                        )
                    ],
                )

    print(f"[output] runs: {args.run_output}", flush=True)
    print(f"[output] summary: {args.summary_output}", flush=True)
    exit_code = 1 if has_failure else 0
    send_bark_notification(args, exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
