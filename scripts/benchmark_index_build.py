#!/usr/bin/env python3
r"""Compare InMemoryIndex::Build time and Linux RSS in fresh processes.

Run on Centaurus after building both static libraries with matching Release
flags. Supply each library's matching include directory; the runner compiles
the same adjacent .cc source twice. Example:

  python3 scripts/benchmark_index_build.py \
    --before-include BASELINE/src/qalsh/include \
    --before-library BASELINE_BUILD/src/qalsh/libqalsh4c_qalsh.a \
    --after-include src/qalsh/include \
    --after-library build/perf-after/src/qalsh/libqalsh4c_qalsh.a \
    --dataset gmm128=build/perf-data/gmm128 --threads 1 8 --norms 1 2 \
    --output build/perf-evidence/index-build.csv

Only Build is timed. Input loading is excluded. RSS includes the input and
index plus allocator-retained pages; no malloc_trim or cache flushing occurs.
Peak RSS is process lifetime high water, so prebuild peak is also recorded.
Parameters use the public library derivation at approximation ratio 2, with
seed 42 by default. No synthetic coordinate generation is involved.
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
import shlex
import statistics
import subprocess
import sys


METRICS = ['build_ms', 'prebuild_rss_kib', 'steady_rss_kib', 'rss_delta_kib',
           'prebuild_peak_rss_kib', 'peak_rss_kib']
CONFIG = ['points', 'dimensions', 'norm', 'threads', 'seed', 'input_bytes',
          'tables', 'collision_threshold']


def digest(path: Path) -> str:
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    for variant in ['before', 'after']:
        parser.add_argument(f'--{variant}-include', type=Path, required=True)
        parser.add_argument(f'--{variant}-library', type=Path, required=True)
    parser.add_argument('--dataset', action='append', required=True, help='LABEL=canonical/dataset/path')
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 8])
    parser.add_argument('--norms', type=int, nargs='+', choices=[1, 2], default=[1, 2])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--cxx', default=os.environ.get('CXX', 'c++'))
    parser.add_argument('--cxxflags', default='-O3 -DNDEBUG -march=native -g',
                        help='same probe flags for both builds; use --cxxflags=... for a leading dash')
    parser.add_argument('--work-dir', type=Path, default=Path('build/index-build-benchmark'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('run this benchmark on Linux (Centaurus)')
    if (args.repeats < 1 or args.warmups < 0 or min(args.threads) < 1 or
            max(args.threads) > 1024 or not 0 <= args.seed < 2**32):
        parser.error('invalid repeat, thread count, or seed')
    compiler = shlex.split(args.cxx)
    if not compiler:
        parser.error('--cxx must name a compiler')
    args.threads = list(dict.fromkeys(args.threads))
    args.norms = list(dict.fromkeys(args.norms))
    variants = {}
    for variant in ['before', 'after']:
        include = getattr(args, variant + '_include').resolve()
        library = getattr(args, variant + '_library').resolve()
        header = include / 'qalsh/qalsh.h'
        if not header.is_file() or not library.is_file():
            parser.error(f'{variant}: matching qalsh/qalsh.h and static library are required')
        variants[variant] = {'include': include, 'library': library, 'header': header}
    datasets = {}
    for value in args.dataset:
        if '=' not in value:
            parser.error('--dataset requires LABEL=path')
        label, directory = value.split('=', 1)
        if (not label or Path(label).name != label or label in ('.', '..') or
                label in datasets or not directory):
            parser.error('dataset labels must be unique basenames with a nonempty path')
        directory = Path(directory).resolve()
        try:
            metadata = json.loads((directory / 'metadata.json').read_text())
            count, dimensions = metadata['num_points']['b'], metadata['num_dimensions']
            if type(count) is not int or not 1 <= count <= 2**30:
                raise ValueError('num_points.b must be an integer in [1, 2^30]')
            if type(dimensions) is not int or not 1 <= dimensions <= 2**20:
                raise ValueError('num_dimensions must be an integer in [1, 2^20]')
            if (directory / 'B.bin').stat().st_size != count * dimensions * 4:
                raise ValueError('B.bin byte size does not match canonical float32 metadata')
        except (OSError, KeyError, TypeError, ValueError) as error:
            parser.error(f'{directory}: {error}')
        datasets[label] = (directory, metadata)

    source = Path(__file__).with_suffix('.cc').resolve()
    args.work_dir = args.work_dir.resolve()
    args.output = args.output.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(str(args.output) + '.json')
    env = {**os.environ, 'OMP_PROC_BIND': 'false', 'OMP_DYNAMIC': 'false'}
    manifest = {
        'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
        'host': platform.uname()._asdict(), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'source': {'path': str(source), 'sha256': digest(source)},
        'warmups': args.warmups, 'repeats': args.repeats, 'variants': {}, 'inputs': {},
        'measurement': 'one Build per process; input preloaded; smaps_rollup Rss; getrusage ru_maxrss; no trim',
        'parameters': 'DeriveQalshParameters(metric, B count, approximation_ratio=2); default budget/error',
        'environment': {'OMP_PROC_BIND': 'false', 'OMP_DYNAMIC': 'false'},
    }
    try:
        manifest['compiler_version'] = subprocess.run(
            compiler + ['--version'], check=True, capture_output=True, text=True).stdout
        for variant, paths in variants.items():
            binary = args.work_dir / ('index-build-' + variant)
            command = compiler + ['-std=c++20', '-pthread'] + shlex.split(args.cxxflags) + [
                '-I', str(paths['include']), str(source), str(paths['library']), '-o', str(binary)]
            record = {name: {'path': str(path), 'sha256': digest(path)}
                      for name, path in paths.items() if name != 'include'}
            record['compile_command'] = command
            manifest['variants'][variant] = record
            subprocess.run(command, check=True)
            record['binary'] = {'path': str(binary), 'sha256': digest(binary)}
            paths['binary'] = binary
        for label, (directory, metadata) in datasets.items():
            manifest['inputs'][label] = {'path': str(directory), 'metadata': metadata,
                'sha256': {name: digest(directory / name) for name in ['B.bin', 'metadata.json']}}
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        with args.output.open('w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=['dataset', 'repeat', 'variant'] + CONFIG + METRICS + ['command'])
            writer.writeheader()
            for label, (directory, metadata) in datasets.items():
                for norm in args.norms:
                    for threads in args.threads:
                        reference = None
                        cases = {'before': [], 'after': []}
                        for repeat in range(-args.warmups, args.repeats):
                            order = ['before', 'after'] if repeat % 2 == 0 else ['after', 'before']
                            for variant in order:
                                command = [str(variants[variant]['binary']), str(directory / 'B.bin'),
                                    str(metadata['num_points']['b']), str(metadata['num_dimensions']),
                                    str(norm), str(threads), str(args.seed)]
                                result = subprocess.run(command, env=env, capture_output=True, text=True)
                                if result.returncode:
                                    raise RuntimeError(f'{command}\n{result.stdout}\n{result.stderr}')
                                row = json.loads(result.stdout)
                                config = {key: row[key] for key in CONFIG}
                                if reference is None:
                                    reference = config
                                if config != reference:
                                    raise RuntimeError(f'{label}: build parameters changed between runs: {config}')
                                if not math.isfinite(row['build_ms']) or row['build_ms'] < 0:
                                    raise RuntimeError(f'{label}: invalid build duration')
                                if repeat < 0:
                                    continue
                                writer.writerow({'dataset': label, 'repeat': repeat, 'variant': variant,
                                                 **row, 'command': json.dumps(command)})
                                output.flush()
                                cases[variant].append(row)
                        summaries = {variant: {key: statistics.median(row[key] for row in rows)
                                              for key in ['build_ms', 'steady_rss_kib', 'peak_rss_kib']}
                                     for variant, rows in cases.items()}
                        print(f'{label} L{norm} T{threads}: ' + json.dumps(summaries), flush=True)
        manifest['status'] = 'complete'
    except BaseException as error:
        manifest['status'] = 'failed'
        manifest['error'] = str(error)
        raise
    finally:
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
