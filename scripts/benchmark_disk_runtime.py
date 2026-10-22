#!/usr/bin/env python3
"""Compare persistent index builds and disk QALSH sampling on Centaurus.

Inputs use canonical float32 A.bin/B.bin files. Each iteration rebuilds the
index and regenerates weights in isolated fixtures. OS caches stay warm;
hashing/validation and fixture preparation are outside process timings.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import tempfile
import time


INDEX_FIELDS = ('approximation_ratio', 'bucket_width', 'collision_threshold',
                'layout', 'num_dimensions', 'num_hash_tables', 'num_points', 'seed')
CSV_FIELDS = ['dataset', 'norm', 'threads', 'layout', 'samples', 'index_seed',
              'repeat', 'variant', 'phase', 'wall_ms', 'estimator_ms', 'estimate',
              'index_sha256', 'metadata_sha256', 'weight_sha256', 'command']


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def timed_run(command: list[str], env: dict[str, str]) -> tuple[float, str]:
    start = time.perf_counter()
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    elapsed = (time.perf_counter() - start) * 1000
    if result.returncode:
        raise RuntimeError(f'command failed ({result.returncode}): {json.dumps(command)}\n'
                           f'{result.stdout}\n{result.stderr}')
    return elapsed, result.stdout


def output_value(output: str, label: str, suffix: str = '') -> str:
    match = re.search(r'^' + re.escape(label) + r':\s*([^\s]+)' +
                      re.escape(suffix) + r'\s*$', output, re.M)
    if not match or not math.isfinite(float(match[1])):
        raise RuntimeError(f'missing or non-finite {label}:\n{output}')
    return match[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--dataset', action='append', required=True,
                        help='LABEL=canonical/dataset/path; repeat for more inputs')
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 8])
    parser.add_argument('--norms', type=int, nargs='+', choices=[1, 2], default=[1, 2])
    parser.add_argument('--layouts', nargs='+', choices=['b-plus-tree', 'sorted-array'],
                        default=['b-plus-tree', 'sorted-array'])
    parser.add_argument('--samples', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1608637542,
                        help='Index seed; default matches fixed-seed in-memory QALSH')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--work-dir', '--workdir', type=Path, dest='work_dir',
                        default=Path('build/disk-runtime-benchmark'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (args.repeats < 1 or args.warmups < 0 or not 1 <= args.samples < 2**32 or
            min(args.threads) < 1 or max(args.threads) > 1024 or not 0 <= args.seed < 2**32):
        parser.error('invalid repeat, sample, thread count or seed')
    for name in ['threads', 'norms', 'layouts']:
        setattr(args, name, list(dict.fromkeys(getattr(args, name))))
    args.binaries = {name: getattr(args, name).resolve() for name in ['before', 'after']}
    for binary in args.binaries.values():
        if not binary.is_file() or not os.access(binary, os.X_OK):
            parser.error(f'missing executable: {binary}')
    args.datasets = {}
    for value in args.dataset:
        if '=' not in value:
            parser.error('--dataset requires LABEL=path')
        label, source_text = value.split('=', 1)
        if (not label or Path(label).name != label or label in ('.', '..') or
                label in args.datasets or not source_text):
            parser.error('dataset labels must be unique directory basenames with a nonempty path')
        source = Path(source_text).resolve()
        try:
            metadata = json.loads((source / 'metadata.json').read_text())
            dimensions = metadata['num_dimensions']
            if type(dimensions) is not int or dimensions < 1:
                raise ValueError('num_dimensions must be a positive integer')
            for side in ['a', 'b']:
                count = metadata['num_points'][side]
                if type(count) is not int or count < 1:
                    raise ValueError(f'num_points.{side} must be a positive integer')
                expected = count * dimensions * 4
                if (source / f'{side.upper()}.bin').stat().st_size != expected:
                    raise ValueError(f'expected float32 {side.upper()}.bin ({expected} bytes)')
        except (OSError, KeyError, TypeError, ValueError) as error:
            parser.error(f'{source}: {error}')
        args.datasets[label] = (source, metadata)
    return args


def main() -> None:
    args = parse_args()
    args.work_dir = args.work_dir.resolve()
    args.output = args.output.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = (args.output.with_suffix('.json') if args.output.suffix == '.csv'
                     else Path(str(args.output) + '.json'))
    env = {**os.environ, 'OMP_PROC_BIND': 'false', 'OMP_DYNAMIC': 'false'}
    manifest = {
        'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
        'script_sha256': digest(Path(__file__)),
        'binaries': {key: {'path': str(path), 'sha256': digest(path)}
                     for key, path in args.binaries.items()},
        'inputs': {}, 'cases': [], 'threads': args.threads, 'norms': args.norms,
        'layouts': args.layouts, 'samples': args.samples, 'index_seed': args.seed,
        'sampling_seed': 42, 'direction': 'ab', 'approximation_ratio': 2,
        'disk_query_batch_size': 64, 'disk_block_size': 4096,
        'warmups': args.warmups, 'repeats': args.repeats,
        'cache_policy': 'warm OS cache; every index rebuilt and weight cache regenerated',
        'timing_policy': 'process wall time; total is paired build + sampling; excludes validation',
        'estimate_check': 'T=1: identical displayed value; T>1: relative tolerance 2e-6, absolute 0',
        'index_timing': 'index CLI has no internal timer; estimator_ms is blank for build rows',
        'environment': {key: value for key, value in env.items()
                        if key.startswith(('OMP_', 'GOMP_'))},
        'host': {'platform': platform.platform(), 'python': platform.python_version(),
                 'logical_cpus': os.cpu_count(),
                 'affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None},
    }
    for label, (source, metadata) in args.datasets.items():
        manifest['inputs'][label] = {
            'path': str(source), 'metadata': metadata,
            'sha256': {name: digest(source / name) for name in ['A.bin', 'B.bin', 'metadata.json']},
        }
    def save_manifest() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    save_manifest()
    try:
        with args.output.open('w', newline='') as output, tempfile.TemporaryDirectory(dir=args.work_dir) as temp:
            writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for label, (source, metadata) in args.datasets.items():
                fixtures = {}
                for variant in args.binaries:
                    fixture = Path(temp) / label / variant
                    fixture.mkdir(parents=True)
                    for name in ['A.bin', 'B.bin']:
                        try: os.link(source / name, fixture / name)
                        except OSError: shutil.copyfile(source / name, fixture / name)
                    shutil.copyfile(source / 'metadata.json', fixture / 'metadata.json')
                    fixtures[variant] = fixture
                for norm in args.norms:
                    for threads in args.threads:
                        case_env = {**env, 'OMP_NUM_THREADS': str(threads)}
                        for layout in args.layouts:
                            reference = {}
                            times = {key: {phase: [] for phase in ['build', 'sampling', 'total']}
                                     for key in args.binaries}
                            case_label = f'{label} L{norm} T{threads} {layout}'
                            for repeat in range(-args.warmups, args.repeats):
                                order = ['before', 'after'] if repeat % 2 == 0 else ['after', 'before']
                                for variant in order:
                                    fixture = fixtures[variant]
                                    prefix = [str(args.binaries[variant]), '-T', str(threads), '--use-fixed-seed']
                                    build_command = prefix + ['index', 'qalsh', '-p', str(norm), '-d', str(fixture),
                                        '--target', 'b', '--seed', str(args.seed), '-c', '2',
                                        '--layout', layout, '--overwrite']
                                    build_ms, _ = timed_run(build_command, case_env)
                                    index_dir = fixture / 'index' / 'qalsh' / f'l{norm}' / 'b'
                                    index_hash = digest(index_dir / 'index.qalsh')
                                    index_metadata = json.loads((index_dir / 'meta.json').read_text())
                                    selected = {key: index_metadata[key] for key in INDEX_FIELDS}
                                    expected = {'layout': layout, 'seed': args.seed, 'approximation_ratio': 2,
                                                'num_points': metadata['num_points']['b'],
                                                'num_dimensions': metadata['num_dimensions']}
                                    if any(selected[key] != value for key, value in expected.items()):
                                        raise RuntimeError(f'{case_label}: unexpected index metadata: {selected}')
                                    metadata_hash = digest(index_dir / 'meta.json')
                                    if not reference:
                                        reference.update(index_sha256=index_hash, index_metadata=selected)
                                    if index_hash != reference['index_sha256'] or selected != reference['index_metadata']:
                                        raise RuntimeError(f'{case_label}: persisted index bytes or metadata changed')
                                    common = dict(dataset=label, norm=norm, threads=threads, layout=layout,
                                        samples=args.samples, index_seed=args.seed, repeat=repeat, variant=variant,
                                        index_sha256=index_hash, metadata_sha256=metadata_hash)
                                    if repeat >= 0:
                                        writer.writerow({**common, 'phase': 'build', 'wall_ms': f'{build_ms:.3f}',
                                                         'command': json.dumps(build_command)})
                                        output.flush()
                                    weight_path = fixture / f'l{norm}_qalsh_weights_A.bin'
                                    weight_path.unlink(missing_ok=True)
                                    query_command = prefix + ['estimate', '--disk-based', '--disk-query-batch-size',
                                        '64', '--disk-block-size', '4096', '-p', str(norm), '-d', str(fixture),
                                        '--direction', 'ab', 'sampling', '--use-cache', '-n', str(args.samples),
                                        'qalsh', '-c', '2']
                                    query_ms, stdout = timed_run(query_command, case_env)
                                    estimate = output_value(stdout, 'Estimated Distance')
                                    estimator_ms = output_value(stdout, 'Time Consumed', ' ms')
                                    if weight_path.stat().st_size != metadata['num_points']['a'] * 4:
                                        raise RuntimeError(f'{case_label}: unexpected weight file size')
                                    weight_hash = digest(weight_path)
                                    if 'weight_sha256' not in reference:
                                        reference.update(weight_sha256=weight_hash, estimate=estimate)
                                    if weight_hash != reference['weight_sha256']:
                                        raise RuntimeError(f'{case_label}: weight bytes changed')
                                    estimates_match = (estimate == reference['estimate'] if threads == 1 else
                                        math.isclose(float(estimate), float(reference['estimate']),
                                                     rel_tol=2e-6, abs_tol=0.0))
                                    if not estimates_match:
                                        raise RuntimeError(f'{case_label}: estimate changed: '
                                                           f'{reference["estimate"]} -> {estimate}')
                                    if repeat < 0: continue
                                    writer.writerow({**common, 'phase': 'sampling', 'wall_ms': f'{query_ms:.3f}',
                                        'estimator_ms': estimator_ms, 'estimate': estimate,
                                        'weight_sha256': weight_hash, 'command': json.dumps(query_command)})
                                    output.flush()
                                    times[variant]['build'].append(build_ms)
                                    times[variant]['sampling'].append(query_ms)
                                    times[variant]['total'].append(build_ms + query_ms)
                            medians = {phase: {variant: statistics.median(times[variant][phase])
                                               for variant in args.binaries}
                                       for phase in ['build', 'sampling', 'total']}
                            manifest['cases'].append(dict(dataset=label, norm=norm, threads=threads,
                                                          layout=layout, **reference, medians_ms=medians))
                            save_manifest()
                            for phase, values in medians.items():
                                print(f'{case_label} {phase}: {values["before"]:.2f} -> '
                                      f'{values["after"]:.2f} ms '
                                      f'({values["before"] / values["after"]:.3f}x)', flush=True)
    except BaseException as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        save_manifest()
        raise
    manifest.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat())
    save_manifest()


if __name__ == '__main__':
    main()
