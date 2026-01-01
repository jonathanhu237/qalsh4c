#!/usr/bin/env python3
"""Experiment 4: compare in-memory QALSH sampling and quadtree sampling."""

from __future__ import annotations

import argparse
import statistics
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import experiment_utils as _experiment_utils

# Current qalsh4c datasets store Coordinate=float.
_experiment_utils.BYTES_PER_COORDINATE = 4

from experiment_utils import (
    RUN_FIELDS,
    SUMMARY_FIELDS,
    build_sampling_command,
    cache_norm_token,
    common_run_row,
    configure_process_limits,
    csv_writer,
    ensure_success,
    finalize_common_args,
    make_cases,
    maybe_build_release,
    run_or_print,
    summary_from_rows,
    to_float,
    validate_cases,
    weight_cache_paths,
    write_rows,
)


EXPERIMENT = "experiment4_qalsh_default_vs_quadtree_t100"
DEFAULT_BARK_BASE_URL = ""
DEFAULT_DATASETS = ["federalist", "mnist", "trevi", "gist"]
RUN_FIELDS_LOCAL = RUN_FIELDS + ["quadtree_max_levels", "measurement_role"]
SUMMARY_FIELDS_LOCAL = SUMMARY_FIELDS + [
    "quadtree_max_levels",
    "weight_generation_time_ms",
    "weight_generation_wall_ms",
    "sampling_mean_time_ms",
    "sampling_stdev_time_ms",
    "sampling_mean_wall_ms",
    "sampling_stdev_wall_ms",
    "speed_total_time_ms",
    "speed_total_wall_ms",
    "weight_generation_cold_probe_time_ms",
    "weight_generation_cached_probe_time_ms",
    "weight_generation_cold_probe_wall_ms",
    "weight_generation_cached_probe_wall_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare in-memory QALSH sampling and quadtree sampling on Federalist, MNIST, "
            "and Trevi. QALSH sampling keeps its automatic sample count; quadtree sampling "
            "uses a fixed sample count of 100 by default. Accuracy is averaged over cached "
            "sampling runs after one weight-generation pass."
        )
    )
    _experiment_utils.add_common_arguments(parser)
    parser.set_defaults(datasets=DEFAULT_DATASETS)
    parser.add_argument("--runs", type=int, default=20, help="Cached sampling runs for accuracy and sampling time.")
    parser.add_argument("--delta", default="0.1", help="Failure probability for automatic QALSH sample count.")
    parser.add_argument("--epsilon", default="1.0", help="Epsilon for automatic QALSH sample count.")
    parser.add_argument("--max-levels", type=int, default=3, help="Maximum quadtree levels; 0 splits until singleton.")
    parser.add_argument("--quadtree-samples", type=int, default=100, help="Fixed quadtree sampling count.")
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
    if args.max_levels < 0:
        raise SystemExit("--max-levels must be non-negative")
    if args.quadtree_samples < 1:
        raise SystemExit("--quadtree-samples must be at least 1")
    if args.bark_timeout <= 0.0:
        raise SystemExit("--bark-timeout must be positive")
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
    title = urllib.parse.quote(f"QALSH4C Exp4 {status}", safe="")
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


def sources_for_direction(direction: str) -> list[str]:
    sources: list[str] = []
    if direction in ("ab", "both"):
        sources.append("A")
    if direction in ("ba", "both"):
        sources.append("B")
    return sources


def quadtree_weight_cache_paths(case, direction: str, max_levels: int) -> list[Path]:
    norm_token = cache_norm_token(case.norm_order)
    return [
        case.dataset_dir / f"l{norm_token}_quadtree_l{max_levels}_weights_{source}.bin"
        for source in sources_for_direction(direction)
    ]


def build_quadtree_command(
    args: argparse.Namespace,
    case,
    num_samples: int | None,
    *,
    use_fixed_seed: bool = False,
) -> list[str]:
    command = build_sampling_command(
        args,
        case,
        "quadtree",
        num_samples=num_samples,
        epsilon=args.epsilon,
        use_cache=True,
        use_fixed_seed=use_fixed_seed,
    )
    command.extend(["--max-levels", str(args.max_levels)])
    return command


def build_algorithm_command(
    args: argparse.Namespace,
    case,
    algorithm: str,
    *,
    num_samples: int | None,
    use_fixed_seed: bool = False,
) -> list[str]:
    if algorithm == "quadtree":
        return build_quadtree_command(args, case, num_samples, use_fixed_seed=use_fixed_seed)
    return build_sampling_command(
        args,
        case,
        algorithm,
        num_samples=num_samples,
        epsilon=args.epsilon,
        use_cache=True,
        use_fixed_seed=use_fixed_seed,
    )


def remove_weight_caches(cache_paths: list[Path], label: str, *, dry_run: bool = False) -> None:
    for cache_path in cache_paths:
        if dry_run:
            print(f"[weights] would remove {label} cache {cache_path}", flush=True)
            continue
        try:
            cache_path.unlink()
            print(f"[weights] removed {label} cache {cache_path}", flush=True)
        except FileNotFoundError:
            pass


def parse_time_ms(row: dict[str, str]) -> float | None:
    return to_float(row["time_ms"])


def parse_wall_ms(row: dict[str, str]) -> float | None:
    return to_float(row["wall_ms"])


def format_ms(value: float | None) -> str:
    return f"{value:.12g}" if value is not None else ""


def mean_text(values: list[float]) -> str:
    return f"{statistics.fmean(values):.12g}" if values else ""


def mean_value(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def stdev_value(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def estimate_generation_value(cold_row: dict[str, str], cached_probe_row: dict[str, str], field: str) -> float | None:
    cold_value = to_float(cold_row[field])
    cached_value = to_float(cached_probe_row[field])
    if cold_value is None:
        return None
    if cached_value is None:
        return cold_value
    return max(0.0, cold_value - cached_value)


def add_local_fields(
    row: dict[str, str],
    *,
    quadtree_max_levels: str = "",
    measurement_role: str = "",
) -> dict[str, str]:
    row["quadtree_max_levels"] = quadtree_max_levels
    row["measurement_role"] = measurement_role
    return row


def add_cache_paths(row: dict[str, str], cache_paths: list[Path]) -> dict[str, str]:
    row["weight_cache_paths"] = "|".join(str(path) for path in cache_paths)
    return row


def make_run_row(
    *,
    args: argparse.Namespace,
    case,
    algorithm: str,
    summary_algorithm: str,
    run_number: int,
    command: list[str],
    result,
    parameter_value: str,
    sample_count_requested: str,
    cache_paths: list[Path],
    measurement_role: str,
) -> dict[str, str]:
    row = common_run_row(
        args=args,
        experiment=EXPERIMENT,
        case=case,
        algorithm=summary_algorithm,
        run_number=run_number,
        command=command,
        result=result,
        parameter_name="sample_count",
        parameter_value=parameter_value,
        epsilon=args.epsilon if algorithm == "qalsh" else "",
        delta=str(args.delta) if algorithm == "qalsh" else "",
        sample_count_requested=sample_count_requested,
    )
    add_cache_paths(row, cache_paths)
    return add_local_fields(
        row,
        quadtree_max_levels=str(args.max_levels) if algorithm == "quadtree" else "",
        measurement_role=measurement_role,
    )


def run_probe_pair(
    args: argparse.Namespace,
    case,
    algorithm: str,
    summary_algorithm: str,
    cache_paths: list[Path],
) -> tuple[list[dict[str, str]], bool]:
    cold_command = build_algorithm_command(args, case, algorithm, num_samples=1)
    cold_label = f"{case.dataset} p={case.norm_order} {algorithm} cold weight probe T=1"
    cold_result = run_or_print(args, cold_command, cold_label)
    cold_ok = ensure_success(cold_result, cold_label)
    cold_row = make_run_row(
        args=args,
        case=case,
        algorithm=algorithm,
        summary_algorithm=summary_algorithm,
        run_number=0,
        command=cold_command,
        result=cold_result,
        parameter_value="1",
        sample_count_requested="1",
        cache_paths=cache_paths,
        measurement_role="weight_generation_cold_probe",
    )

    cached_command = build_algorithm_command(args, case, algorithm, num_samples=1)
    cached_label = f"{case.dataset} p={case.norm_order} {algorithm} cached weight probe T=1"
    cached_result = run_or_print(args, cached_command, cached_label)
    cached_ok = ensure_success(cached_result, cached_label)
    cached_row = make_run_row(
        args=args,
        case=case,
        algorithm=algorithm,
        summary_algorithm=summary_algorithm,
        run_number=0,
        command=cached_command,
        result=cached_result,
        parameter_value="1",
        sample_count_requested="1",
        cache_paths=cache_paths,
        measurement_role="weight_generation_cached_probe",
    )
    return [cold_row, cached_row], cold_ok and cached_ok


def run_sampling_rows(
    args: argparse.Namespace,
    case,
    algorithm: str,
    summary_algorithm: str,
    sample_count: int | None,
    cache_paths: list[Path],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    parameter_value = "auto" if sample_count is None else str(sample_count)
    for run_number in range(1, args.runs + 1):
        command = build_algorithm_command(args, case, algorithm, num_samples=sample_count)
        label = f"{case.dataset} p={case.norm_order} {algorithm} T={parameter_value} run={run_number}"
        result = run_or_print(args, command, label)
        ensure_success(result, label)
        rows.append(
            make_run_row(
                args=args,
                case=case,
                algorithm=algorithm,
                summary_algorithm=summary_algorithm,
                run_number=run_number,
                command=command,
                result=result,
                parameter_value=parameter_value,
                sample_count_requested=parameter_value,
                cache_paths=cache_paths,
                measurement_role="cached_sampling",
            )
        )
    return rows


def make_summary(
    *,
    args: argparse.Namespace,
    case,
    algorithm: str,
    rows: list[dict[str, str]],
    runs_requested: int,
    parameter_name: str,
    parameter_value: str,
    epsilon: str,
    delta: str,
    sample_count_requested: str,
    quadtree_max_levels: str,
    cold_probe_row: dict[str, str],
    cached_probe_row: dict[str, str],
) -> dict[str, str]:
    sampling_time_values = [value for row in rows if (value := parse_time_ms(row)) is not None]
    sampling_wall_values = [value for row in rows if (value := parse_wall_ms(row)) is not None]
    sampling_mean_time = mean_value(sampling_time_values)
    sampling_mean_wall = mean_value(sampling_wall_values)
    generation_time = estimate_generation_value(cold_probe_row, cached_probe_row, "time_ms")
    generation_wall = estimate_generation_value(cold_probe_row, cached_probe_row, "wall_ms")
    speed_total_time = None if generation_time is None or sampling_mean_time is None else generation_time + sampling_mean_time
    speed_total_wall = None if generation_wall is None or sampling_mean_wall is None else generation_wall + sampling_mean_wall

    summary = summary_from_rows(
        args=args,
        experiment=EXPERIMENT,
        case=case,
        algorithm=algorithm,
        rows=rows,
        runs_requested=runs_requested,
        run_output=args.run_output,
        parameter_name=parameter_name,
        parameter_value=parameter_value,
        epsilon=epsilon,
        delta=delta,
        sample_count_requested=sample_count_requested,
        time_source="estimated_weight_generation_plus_cached_sampling_mean",
    )
    summary["quadtree_max_levels"] = quadtree_max_levels
    summary["weight_generation_time_ms"] = format_ms(generation_time)
    summary["weight_generation_wall_ms"] = format_ms(generation_wall)
    summary["sampling_mean_time_ms"] = mean_text(sampling_time_values)
    summary["sampling_stdev_time_ms"] = format_ms(stdev_value(sampling_time_values))
    summary["sampling_mean_wall_ms"] = mean_text(sampling_wall_values)
    summary["sampling_stdev_wall_ms"] = format_ms(stdev_value(sampling_wall_values))
    summary["speed_total_time_ms"] = format_ms(speed_total_time)
    summary["speed_total_wall_ms"] = format_ms(speed_total_wall)
    summary["weight_generation_cold_probe_time_ms"] = cold_probe_row["time_ms"]
    summary["weight_generation_cached_probe_time_ms"] = cached_probe_row["time_ms"]
    summary["weight_generation_cold_probe_wall_ms"] = cold_probe_row["wall_ms"]
    summary["weight_generation_cached_probe_wall_ms"] = cached_probe_row["wall_ms"]
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

    run_writer, run_file = csv_writer(args.run_output, args.append, RUN_FIELDS_LOCAL)
    summary_writer, summary_file = csv_writer(args.summary_output, args.append, SUMMARY_FIELDS_LOCAL)
    has_failure = False

    with run_file, summary_file:
        for case in cases:
            algorithms = [
                ("qalsh", "qalsh_sampling", None, weight_cache_paths(case, args.direction), ""),
                (
                    "quadtree",
                    "quadtree_sampling",
                    args.quadtree_samples,
                    quadtree_weight_cache_paths(case, args.direction, args.max_levels),
                    str(args.max_levels),
                ),
            ]

            for qalsh4c_algorithm, summary_algorithm, sample_count, cache_paths, quadtree_levels in algorithms:
                remove_weight_caches(cache_paths, qalsh4c_algorithm, dry_run=args.dry_run)

                probe_rows, probes_ok = run_probe_pair(
                    args,
                    case,
                    qalsh4c_algorithm,
                    summary_algorithm,
                    cache_paths,
                )
                write_rows(run_writer, run_file, probe_rows)
                if not probes_ok:
                    has_failure = True
                    continue

                sampling_rows = run_sampling_rows(
                    args,
                    case,
                    qalsh4c_algorithm,
                    summary_algorithm,
                    sample_count,
                    cache_paths,
                )
                if any(row["return_code"] != "0" for row in sampling_rows):
                    has_failure = True
                write_rows(run_writer, run_file, sampling_rows)

                parameter_value = "auto" if sample_count is None else str(sample_count)
                write_rows(
                    summary_writer,
                    summary_file,
                    [
                        make_summary(
                            args=args,
                            case=case,
                            algorithm=summary_algorithm,
                            rows=sampling_rows,
                            runs_requested=args.runs,
                            parameter_name="sample_count",
                            parameter_value=parameter_value,
                            epsilon=args.epsilon if qalsh4c_algorithm == "qalsh" else "",
                            delta=str(args.delta) if qalsh4c_algorithm == "qalsh" else "",
                            sample_count_requested=parameter_value,
                            quadtree_max_levels=quadtree_levels,
                            cold_probe_row=probe_rows[0],
                            cached_probe_row=probe_rows[1],
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
