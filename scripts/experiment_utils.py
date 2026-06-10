#!/usr/bin/env python3
"""Shared helpers for qalsh4c paper experiment scripts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Iterable

try:
    import resource
except ImportError:  # pragma: no cover - resource is available on Linux/macOS.
    resource = None


REPO_ROOT = Path(__file__).resolve().parents[1]
BYTES_PER_COORDINATE = 8

DATASET_LABELS = {
    "federalist": "Federalist",
    "mnist": "Mnist",
    "trevi": "Trevi",
    "gist": "Gist",
}

QALSH_PATTERNS = {
    "time_ms": re.compile(r"^Time Consumed:\s*([0-9]+(?:\.[0-9]+)?)\s*ms\s*$", re.MULTILINE),
    "memory_mb": re.compile(r"^Memory Usage:\s*([-+]?[0-9]+(?:\.[0-9]+)?)\s*MB\s*$", re.MULTILINE),
    "estimated_distance": re.compile(r"^Estimated Distance:\s*([^\n]+?)\s*$", re.MULTILINE),
    "ground_truth": re.compile(r"^Ground Truth:\s*([^\n]+?)\s*$", re.MULTILINE),
    "relative_error_pct": re.compile(r"^Relative Error:\s*([^\n]+?)\s*$", re.MULTILINE),
    "auto_samples": re.compile(r"SamplingEstimator automatically selected\s+([0-9]+)\s+samples\."),
    "default_samples": re.compile(r"Using default sample count:\s+([0-9]+)"),
    "ab_samples": re.compile(r"Using specified A->B sample count:\s+([0-9]+)"),
    "ba_samples": re.compile(r"Using specified B->A sample count:\s+([0-9]+)"),
    "auto_exact_batch_size": re.compile(r"automatically selected exact batch size\s+([0-9]+)"),
    "specified_exact_batch_size": re.compile(r"Using specified exact batch size:\s+([0-9]+)"),
}

RUN_FIELDS = [
    "timestamp",
    "experiment",
    "dataset",
    "dataset_dir",
    "norm_order",
    "direction",
    "algorithm",
    "parameter_name",
    "parameter_value",
    "epsilon",
    "delta",
    "sample_count_requested",
    "run",
    "threads",
    "approximation_ratio",
    "selected_samples",
    "selected_exact_batch_size",
    "estimated_distance",
    "ground_truth",
    "relative_error_pct",
    "time_ms",
    "wall_ms",
    "memory_mb",
    "time_source",
    "return_code",
    "command",
    "weight_cache_paths",
    "source_run_csv",
]

SUMMARY_FIELDS = [
    "timestamp",
    "experiment",
    "dataset",
    "dataset_dir",
    "norm_order",
    "direction",
    "algorithm",
    "parameter_name",
    "parameter_value",
    "epsilon",
    "delta",
    "sample_count_requested",
    "runs_requested",
    "runs_succeeded",
    "threads",
    "approximation_ratio",
    "selected_samples",
    "selected_exact_batch_size",
    "ground_truth",
    "mean_estimated_distance",
    "mean_relative_error_pct",
    "stdev_relative_error_pct",
    "min_relative_error_pct",
    "max_relative_error_pct",
    "mean_time_ms",
    "stdev_time_ms",
    "mean_wall_ms",
    "time_source",
    "run_csv",
]


@dataclass(frozen=True)
class ExperimentCase:
    dataset: str
    dataset_dir: Path
    norm_order: str


@dataclass(frozen=True)
class CommandResult:
    return_code: int
    output: str
    wall_ms: float


def timestamp_for_path() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def timestamp_for_row() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def parse_list(values: Iterable[str], valid: set[str] | None, option_name: str) -> list[str]:
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip().lower() for part in value.split(",") if part.strip())

    if valid is not None:
        invalid = [value for value in parsed if value not in valid]
        if invalid:
            valid_values = ", ".join(sorted(valid))
            raise SystemExit(f"{option_name} contains invalid value(s): {', '.join(invalid)}. Valid: {valid_values}")
    if not parsed:
        raise SystemExit(f"{option_name} must contain at least one value.")
    return parsed


def parse_int_list(values: Iterable[str], option_name: str) -> list[int]:
    parsed = [int(value) for value in parse_list(values, None, option_name)]
    invalid = [value for value in parsed if value < 1]
    if invalid:
        raise SystemExit(f"{option_name} values must be positive: {invalid}")
    return parsed


def int_range(start: int, stop: int, step: int, option_prefix: str) -> list[int]:
    if start < 1:
        raise SystemExit(f"{option_prefix}-start must be at least 1")
    if step < 1:
        raise SystemExit(f"{option_prefix}-step must be at least 1")
    if stop < start:
        raise SystemExit(f"{option_prefix}-stop must be greater than or equal to {option_prefix}-start")
    return list(range(start, stop + 1, step))


def decimal_range(start: str, stop: str, step: str) -> list[str]:
    start_value = Decimal(start)
    stop_value = Decimal(stop)
    step_value = Decimal(step)
    if step_value <= 0:
        raise SystemExit("--epsilon-step must be positive")

    values: list[str] = []
    value = start_value
    while value <= stop_value:
        values.append(format(value.normalize(), "f"))
        value += step_value
    if not values:
        raise SystemExit("epsilon range is empty")
    return values


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--binary",
        type=Path,
        default=REPO_ROOT / "build" / "release" / "qalsh4c",
        help="Path to the qalsh4c executable.",
    )
    parser.add_argument("--build", action="store_true", help="Run cmake --workflow --preset release first.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Root containing federalist/mnist/trevi/gist. Defaults to QALSH4C_DATASET_ROOT or ~/datasets/qalsh4c.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_LABELS),
        help="Datasets to run. Supports spaces or comma-separated names.",
    )
    parser.add_argument(
        "--norm-orders",
        nargs="+",
        default=["1", "2"],
        help="Lp norm orders to run. Supports spaces or comma-separated values.",
    )
    parser.add_argument("--direction", default="ab", choices=["ab", "ba", "both"], help="Chamfer direction.")
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Pass -T/--threads to qalsh4c. Omit to use qalsh4c's default maximum thread count.",
    )
    parser.add_argument(
        "--approximation-ratio",
        default="2.0",
        help="Approximation ratio passed to QALSH ANN and QALSH sampling.",
    )
    parser.add_argument(
        "--exact-batch-size",
        type=int,
        default=0,
        help="Stage 2 exact scan batch size for sampling (0 lets qalsh4c choose automatically).",
    )
    parser.add_argument(
        "--refresh-weights",
        action="store_true",
        help="Delete relevant QALSH weight caches before warming them.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Do not warm missing QALSH weight caches before running QALSH sampling commands.",
    )
    parser.add_argument(
        "--fixed-weight-seed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use --use-fixed-seed only while warming QALSH weight caches.",
    )
    parser.add_argument(
        "--fixed-ann-seed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use --use-fixed-seed for deterministic QALSH ANN one-shot runs.",
    )
    parser.add_argument(
        "--ulimit-n",
        type=int,
        default=65535,
        help="Raise RLIMIT_NOFILE soft limit before running commands, when possible.",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results", help="Default output directory.")
    parser.add_argument("--run-output", type=Path, default=None, help="Per-run CSV output path.")
    parser.add_argument("--summary-output", type=Path, default=None, help="Summary CSV output path.")
    parser.add_argument("--append", action="store_true", help="Append to output CSVs instead of overwriting.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing benchmarks.")


def finalize_common_args(args: argparse.Namespace, stem: str) -> None:
    args.binary = normalize_path(args.binary)
    args.output_dir = normalize_path(args.output_dir)
    if args.dataset_root is not None:
        args.dataset_root = normalize_path(args.dataset_root)
    args.datasets = parse_list(args.datasets, set(DATASET_LABELS), "--datasets")
    args.norm_orders = parse_list(args.norm_orders, {"1", "2"}, "--norm-orders")
    if args.threads is not None and args.threads < 1:
        raise SystemExit("--threads must be at least 1")
    if args.exact_batch_size < 0:
        raise SystemExit("--exact-batch-size must be non-negative")
    if args.ulimit_n is not None and args.ulimit_n < 1:
        raise SystemExit("--ulimit-n must be at least 1")

    output_stem = f"{stem}.local.{timestamp_for_path()}"
    if args.run_output is None:
        args.run_output = args.output_dir / f"{output_stem}.runs.csv"
    else:
        args.run_output = normalize_path(args.run_output)
    if args.summary_output is None:
        args.summary_output = args.output_dir / f"{output_stem}.summary.csv"
    else:
        args.summary_output = normalize_path(args.summary_output)


def dataset_search_roots(args: argparse.Namespace) -> list[Path]:
    roots: list[Path] = []
    env_root = os.environ.get("QALSH4C_DATASET_ROOT")
    if args.dataset_root is not None:
        roots.append(args.dataset_root)
    if env_root:
        roots.append(Path(env_root).expanduser())
    roots.extend(
        [
            Path.home() / "datasets" / "qalsh4c",
            Path.home() / "datasets" / "diskann-chamfer",
            Path.home() / "datasets",
            REPO_ROOT / "data",
        ]
    )
    return roots


def read_metadata(dataset_dir: Path) -> dict:
    with (dataset_dir / "metadata.json").open() as metadata_file:
        return json.load(metadata_file)


def dataset_problems(dataset_dir: Path) -> list[str]:
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.is_file():
        return [f"missing metadata: {metadata_path}"]

    try:
        metadata = read_metadata(dataset_dir)
        num_dimensions = int(metadata["num_dimensions"])
        num_points = {
            "A.bin": int(metadata["num_points"]["a"]),
            "B.bin": int(metadata["num_points"]["b"]),
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return [f"invalid metadata: {metadata_path} ({error})"]

    problems: list[str] = []
    for file_name, expected_points in num_points.items():
        data_path = dataset_dir / file_name
        if not data_path.is_file():
            problems.append(f"missing data file: {data_path}")
            continue
        expected_size = expected_points * num_dimensions * BYTES_PER_COORDINATE
        actual_size = data_path.stat().st_size
        if actual_size != expected_size:
            problems.append(f"bad data size: {data_path} has {actual_size} bytes, expected {expected_size}")
    return problems


def resolve_dataset_dir(args: argparse.Namespace, dataset: str) -> Path:
    for root in dataset_search_roots(args):
        candidate = root / dataset
        if not dataset_problems(candidate):
            return candidate
    fallback_root = args.dataset_root if args.dataset_root is not None else Path.home() / "datasets" / "qalsh4c"
    return fallback_root / dataset


def make_cases(args: argparse.Namespace) -> list[ExperimentCase]:
    cases: list[ExperimentCase] = []
    for dataset in args.datasets:
        dataset_dir = resolve_dataset_dir(args, dataset)
        for norm_order in args.norm_orders:
            cases.append(ExperimentCase(dataset=dataset, dataset_dir=dataset_dir, norm_order=norm_order))
    return cases


def validate_cases(args: argparse.Namespace, cases: Iterable[ExperimentCase]) -> None:
    problems: list[str] = []
    if not args.dry_run and not args.build and not args.binary.is_file():
        problems.append(f"missing executable: {args.binary}")
    for case in cases:
        problems.extend(f"{case.dataset}: {problem}" for problem in dataset_problems(case.dataset_dir))
    if problems:
        joined = "\n  - ".join(problems)
        raise SystemExit(f"Cannot run experiment:\n  - {joined}")


def configure_process_limits(args: argparse.Namespace) -> None:
    if args.ulimit_n is None or resource is None:
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = args.ulimit_n
    if hard != resource.RLIM_INFINITY:
        target = min(target, hard)
    if target > soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    effective_soft, effective_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    hard_text = "unlimited" if effective_hard == resource.RLIM_INFINITY else str(effective_hard)
    print(f"[limits] RLIMIT_NOFILE soft={effective_soft} hard={hard_text}", flush=True)


def maybe_build_release(args: argparse.Namespace) -> None:
    if not args.build:
        return
    command = ["cmake", "--workflow", "--preset", "release"]
    print(f"[build] {shlex.join(command)}", flush=True)
    result = run_command(command)
    if result.return_code != 0:
        print(result.output, file=sys.stderr)
        raise SystemExit(result.return_code)


def run_command(command: list[str]) -> CommandResult:
    start = time.perf_counter()
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    wall_ms = (time.perf_counter() - start) * 1000.0
    return CommandResult(return_code=completed.returncode, output=completed.stdout, wall_ms=wall_ms)


def run_or_print(args: argparse.Namespace, command: list[str], label: str) -> CommandResult:
    print(f"[{label}] {shlex.join(command)}", flush=True)
    if args.dry_run:
        return CommandResult(return_code=0, output="", wall_ms=0.0)
    return run_command(command)


def qalsh_prefix(args: argparse.Namespace, use_fixed_seed: bool = False) -> list[str]:
    command = [str(args.binary)]
    if args.threads is not None:
        command.extend(["-T", str(args.threads)])
    if use_fixed_seed:
        command.append("--use-fixed-seed")
    return command


def build_sampling_command(
    args: argparse.Namespace,
    case: ExperimentCase,
    algorithm: str,
    num_samples: int | None = None,
    epsilon: str | None = None,
    use_cache: bool = False,
    use_fixed_seed: bool = False,
) -> list[str]:
    command = qalsh_prefix(args, use_fixed_seed=use_fixed_seed)
    command.extend(["estimate", "-p", case.norm_order, "-d", str(case.dataset_dir), "--direction", args.direction])
    command.append("sampling")
    if num_samples is not None:
        command.extend(["-n", str(num_samples)])
    if epsilon is not None:
        command.extend(["-e", str(epsilon)])
    if hasattr(args, "delta"):
        command.extend(["--delta", str(args.delta)])
    if use_cache:
        command.append("--use-cache")
    if args.exact_batch_size != 0:
        command.extend(["--exact-batch-size", str(args.exact_batch_size)])
    command.append(algorithm)
    if algorithm == "qalsh":
        command.extend(["-c", str(args.approximation_ratio)])
    return command


def build_ann_command(
    args: argparse.Namespace,
    case: ExperimentCase,
    algorithm: str,
    use_fixed_seed: bool = False,
) -> list[str]:
    command = qalsh_prefix(args, use_fixed_seed=use_fixed_seed)
    command.extend(["estimate", "-p", case.norm_order, "-d", str(case.dataset_dir), "--direction", args.direction])
    command.extend(["ann", algorithm])
    if algorithm == "qalsh":
        command.extend(["-c", str(args.approximation_ratio)])
    return command


def cache_norm_token(norm_order: str) -> str:
    return f"{float(norm_order):g}"


def weight_cache_paths(case: ExperimentCase, direction: str) -> list[Path]:
    sources: list[str] = []
    if direction in ("ab", "both"):
        sources.append("A")
    if direction in ("ba", "both"):
        sources.append("B")
    norm_token = cache_norm_token(case.norm_order)
    return [case.dataset_dir / f"l{norm_token}_qalsh_weights_{source}.bin" for source in sources]


def warm_qalsh_weights(args: argparse.Namespace, case: ExperimentCase) -> bool:
    cache_paths = weight_cache_paths(case, args.direction)
    if args.refresh_weights:
        for cache_path in cache_paths:
            try:
                cache_path.unlink()
                print(f"[weights] removed {cache_path}", flush=True)
            except FileNotFoundError:
                pass

    missing = [cache_path for cache_path in cache_paths if not cache_path.is_file()]
    if args.no_warmup or not missing:
        return True

    command = build_sampling_command(
        args,
        case,
        "qalsh",
        num_samples=1,
        epsilon="1.0",
        use_cache=True,
        use_fixed_seed=args.fixed_weight_seed,
    )
    result = run_or_print(args, command, f"warm {DATASET_LABELS[case.dataset]} p={case.norm_order}")
    if result.return_code != 0:
        print(result.output, file=sys.stderr)
        return False

    if args.dry_run:
        return True
    still_missing = [cache_path for cache_path in cache_paths if not cache_path.is_file()]
    if still_missing:
        for cache_path in still_missing:
            print(f"[weights] cache was not created: {cache_path}", file=sys.stderr, flush=True)
        return False
    return True


def parse_qalsh_output(output: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for field in ("time_ms", "memory_mb", "estimated_distance", "ground_truth", "relative_error_pct"):
        match = QALSH_PATTERNS[field].search(output)
        value = match.group(1).strip() if match else ""
        if value.lower() == "unavailable":
            value = ""
        if field == "relative_error_pct" and value.endswith("%"):
            value = value[:-1].strip()
        parsed[field] = value

    for output_field, pattern_names in {
        "selected_samples": ("auto_samples", "ab_samples", "ba_samples", "default_samples"),
        "selected_exact_batch_size": ("auto_exact_batch_size", "specified_exact_batch_size"),
    }.items():
        parsed[output_field] = ""
        for pattern_name in pattern_names:
            match = QALSH_PATTERNS[pattern_name].search(output)
            if match:
                parsed[output_field] = match.group(1)
                break
    return parsed


def common_run_row(
    *,
    args: argparse.Namespace,
    experiment: str,
    case: ExperimentCase,
    algorithm: str,
    run_number: int,
    command: list[str],
    result: CommandResult,
    parameter_name: str = "",
    parameter_value: str = "",
    epsilon: str = "",
    delta: str = "",
    sample_count_requested: str = "",
    time_source: str = "qalsh4c_time_consumed",
    source_run_csv: str = "",
) -> dict[str, str]:
    parsed = parse_qalsh_output(result.output)
    return {
        "timestamp": timestamp_for_row(),
        "experiment": experiment,
        "dataset": DATASET_LABELS[case.dataset],
        "dataset_dir": str(case.dataset_dir),
        "norm_order": case.norm_order,
        "direction": args.direction,
        "algorithm": algorithm,
        "parameter_name": parameter_name,
        "parameter_value": parameter_value,
        "epsilon": epsilon,
        "delta": delta,
        "sample_count_requested": sample_count_requested,
        "run": str(run_number),
        "threads": str(args.threads) if args.threads is not None else "default",
        "approximation_ratio": str(args.approximation_ratio),
        "selected_samples": parsed["selected_samples"],
        "selected_exact_batch_size": parsed["selected_exact_batch_size"],
        "estimated_distance": parsed["estimated_distance"],
        "ground_truth": parsed["ground_truth"],
        "relative_error_pct": parsed["relative_error_pct"],
        "time_ms": parsed["time_ms"] if time_source == "qalsh4c_time_consumed" else f"{result.wall_ms:.3f}",
        "wall_ms": f"{result.wall_ms:.3f}",
        "memory_mb": parsed["memory_mb"],
        "time_source": time_source,
        "return_code": str(result.return_code),
        "command": shlex.join(command),
        "weight_cache_paths": "|".join(str(path) for path in weight_cache_paths(case, args.direction)),
        "source_run_csv": source_run_csv,
    }


def empty_row() -> dict[str, str]:
    return {field: "" for field in RUN_FIELDS}


def to_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def summary_from_rows(
    *,
    args: argparse.Namespace,
    experiment: str,
    case: ExperimentCase,
    algorithm: str,
    rows: list[dict[str, str]],
    runs_requested: int,
    run_output: Path,
    parameter_name: str = "",
    parameter_value: str = "",
    epsilon: str = "",
    delta: str = "",
    sample_count_requested: str = "",
    time_source: str = "",
) -> dict[str, str]:
    successful_rows = [row for row in rows if row["return_code"] == "0"]
    rel_errors = [value for row in successful_rows if (value := to_float(row["relative_error_pct"])) is not None]
    estimates = [value for row in successful_rows if (value := to_float(row["estimated_distance"])) is not None]
    time_values = [value for row in successful_rows if (value := to_float(row["time_ms"])) is not None]
    wall_values = [value for row in successful_rows if (value := to_float(row["wall_ms"])) is not None]
    selected_samples = next((row["selected_samples"] for row in successful_rows if row["selected_samples"]), "")
    selected_exact_batch_size = next(
        (row["selected_exact_batch_size"] for row in successful_rows if row["selected_exact_batch_size"]), ""
    )
    ground_truth = next((row["ground_truth"] for row in successful_rows if row["ground_truth"]), "")
    effective_time_source = time_source or next(
        (row["time_source"] for row in successful_rows if row["time_source"]), ""
    )

    return {
        "timestamp": timestamp_for_row(),
        "experiment": experiment,
        "dataset": DATASET_LABELS[case.dataset],
        "dataset_dir": str(case.dataset_dir),
        "norm_order": case.norm_order,
        "direction": args.direction,
        "algorithm": algorithm,
        "parameter_name": parameter_name,
        "parameter_value": parameter_value,
        "epsilon": epsilon,
        "delta": delta,
        "sample_count_requested": sample_count_requested,
        "runs_requested": str(runs_requested),
        "runs_succeeded": str(len(successful_rows)),
        "threads": str(args.threads) if args.threads is not None else "default",
        "approximation_ratio": str(args.approximation_ratio),
        "selected_samples": selected_samples,
        "selected_exact_batch_size": selected_exact_batch_size,
        "ground_truth": ground_truth,
        "mean_estimated_distance": f"{statistics.fmean(estimates):.12g}" if estimates else "",
        "mean_relative_error_pct": f"{statistics.fmean(rel_errors):.12g}" if rel_errors else "",
        "stdev_relative_error_pct": f"{statistics.stdev(rel_errors):.12g}" if len(rel_errors) >= 2 else "",
        "min_relative_error_pct": f"{min(rel_errors):.12g}" if rel_errors else "",
        "max_relative_error_pct": f"{max(rel_errors):.12g}" if rel_errors else "",
        "mean_time_ms": f"{statistics.fmean(time_values):.12g}" if time_values else "",
        "stdev_time_ms": f"{statistics.stdev(time_values):.12g}" if len(time_values) >= 2 else "",
        "mean_wall_ms": f"{statistics.fmean(wall_values):.12g}" if wall_values else "",
        "time_source": effective_time_source,
        "run_csv": str(run_output),
    }


def csv_writer(path: Path, append: bool, fields: list[str]) -> tuple[csv.DictWriter, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not append or not path.exists() or path.stat().st_size == 0
    csv_file = path.open("a" if append else "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    if should_write_header:
        writer.writeheader()
        csv_file.flush()
    return writer, csv_file


def write_rows(writer: csv.DictWriter, csv_file: object, rows: Iterable[dict[str, str]]) -> None:
    for row in rows:
        writer.writerow(row)
    csv_file.flush()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def ensure_success(result: CommandResult, label: str) -> bool:
    if result.return_code == 0:
        return True
    print(f"[{label}] failed with exit code {result.return_code}", file=sys.stderr)
    print(result.output, file=sys.stderr)
    return False


def add_bakshi_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--bakshi-script",
        type=Path,
        default=Path.home() / "bakshi4c" / "scripts" / "sweep_bakshi4c_sample_count.local.py",
        help="Path to the Bakshi4C sample-count sweep script.",
    )
    parser.add_argument(
        "--bakshi-python",
        type=Path,
        default=None,
        help="Python executable used to run the Bakshi4C script. Defaults to the bakshi4c mamba env when present.",
    )
    parser.add_argument(
        "--bakshi-max-levels",
        type=int,
        default=3,
        help="Bakshi4C max_levels parameter.",
    )
    parser.add_argument("--bakshi-seed", type=int, default=None, help="Optional Bakshi4C base RNG seed.")
    parser.add_argument(
        "--refresh-bakshi-weights",
        action="store_true",
        help="Ask the Bakshi4C script to regenerate its weight cache files.",
    )
    parser.add_argument(
        "--skip-bakshi",
        action="store_true",
        help="Skip Bakshi4C runs. Useful when smoke-testing qalsh4c-only paths.",
    )


def finalize_bakshi_args(args: argparse.Namespace) -> None:
    args.bakshi_script = normalize_path(args.bakshi_script)
    if args.bakshi_python is None:
        for candidate in (
            Path.home() / "miniforge3" / "envs" / "bakshi4c" / "bin" / "python",
            Path.home() / "miniforge3" / "bin" / "python3",
            Path(sys.executable),
        ):
            if candidate.is_file():
                args.bakshi_python = candidate
                break
    else:
        args.bakshi_python = normalize_path(args.bakshi_python)
    if args.bakshi_max_levels < 1:
        raise SystemExit("--bakshi-max-levels must be at least 1")
    if args.direction != "ab" and not args.skip_bakshi:
        raise SystemExit("Bakshi4C wrapper only supports --direction ab")
    if not args.dry_run and not args.skip_bakshi and not args.bakshi_script.is_file():
        raise SystemExit(f"missing Bakshi4C script: {args.bakshi_script}")
    if not args.dry_run and not args.skip_bakshi and not args.bakshi_python.is_file():
        raise SystemExit(f"missing Bakshi4C Python executable: {args.bakshi_python}")


def build_bakshi_command(
    args: argparse.Namespace,
    case: ExperimentCase,
    sample_counts: Iterable[int],
    run_output: Path,
    summary_output: Path,
    runs: int,
) -> list[str]:
    command = [
        str(args.bakshi_python),
        str(args.bakshi_script),
        "--dataset-root",
        str(case.dataset_dir.parent),
        "--datasets",
        case.dataset,
        "--norm-orders",
        case.norm_order,
        "--direction",
        args.direction,
        "--runs",
        str(runs),
        "--sample-counts",
    ]
    command.extend(str(sample_count) for sample_count in sample_counts)
    command.extend(
        [
            "--max-levels",
            str(args.bakshi_max_levels),
            "--run-output",
            str(run_output),
            "--summary-output",
            str(summary_output),
        ]
    )
    if args.bakshi_seed is not None:
        command.extend(["--seed", str(args.bakshi_seed)])
    if args.refresh_bakshi_weights:
        command.append("--refresh-weights")
    if args.dry_run:
        command.append("--dry-run")
    return command


def bakshi_run_row_to_common(
    *,
    args: argparse.Namespace,
    experiment: str,
    case: ExperimentCase,
    source_row: dict[str, str],
    source_run_csv: Path,
) -> dict[str, str]:
    sample_count = source_row.get("sample_count_requested", "")
    row = empty_row()
    row.update(
        {
            "timestamp": source_row.get("timestamp", timestamp_for_row()),
            "experiment": experiment,
            "dataset": DATASET_LABELS[case.dataset],
            "dataset_dir": str(case.dataset_dir),
            "norm_order": case.norm_order,
            "direction": args.direction,
            "algorithm": "bakshi4c",
            "parameter_name": "sample_count",
            "parameter_value": sample_count,
            "sample_count_requested": sample_count,
            "run": source_row.get("run", ""),
            "threads": "python",
            "approximation_ratio": "",
            "selected_samples": source_row.get("selected_samples", sample_count),
            "estimated_distance": source_row.get("estimated_distance", ""),
            "ground_truth": source_row.get("ground_truth", ""),
            "relative_error_pct": source_row.get("relative_error_pct", ""),
            "time_ms": source_row.get("query_wall_ms", ""),
            "wall_ms": source_row.get("query_wall_ms", ""),
            "time_source": "bakshi_wall_clock",
            "return_code": "0" if source_row.get("relative_error_pct", "") else "1",
            "command": source_row.get("command", ""),
            "source_run_csv": str(source_run_csv),
        }
    )
    return row
