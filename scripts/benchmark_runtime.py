#!/usr/bin/env python3
"""Compare complete QALSH sampling runs with isolated, verifiable weight caches.

Run on Centaurus. Inputs must use the current canonical float32 format.
The cold mode regenerates weights (including the in-memory index); cached mode
reuses them. OS caches are warmed; no privileged cache flushing is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import tempfile
import time


def digest(path: Path) -> str:
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--dataset', action='append', required=True, help='LABEL=canonical/dataset/path')
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 8])
    parser.add_argument('--norms', type=int, nargs='+', choices=[1, 2], default=[1, 2])
    parser.add_argument('--samples', type=int, default=512, help='0 selects the automatic sample count')
    parser.add_argument('--modes', nargs='+', choices=['cold', 'cached'], default=['cold', 'cached'])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--work-dir', type=Path, default=Path('build/runtime-benchmark'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0 or args.samples < 0 or min(args.threads) < 1:
        parser.error('invalid repeat, sample or thread count')
    binaries = {key: value.resolve() for key, value in [('before', args.before), ('after', args.after)]}
    for binary in binaries.values():
        if not binary.is_file(): parser.error(f'missing binary: {binary}')
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ['dataset', 'norm', 'threads', 'samples', 'mode', 'repeat', 'variant',
              'wall_ms', 'estimator_ms', 'estimate', 'weight_sha256', 'command']
    manifest = {'binaries': {key: {'path': str(path), 'sha256': digest(path)} for key, path in binaries.items()},
                'inputs': {}, 'cache_policy': 'warm OS cache; cold regenerates weight cache',
                'warmups': args.warmups, 'repeats': args.repeats}
    rows = []
    # Avoid pinning the main OpenMP thread to one CPU: std::jthread build workers
    # inherit its affinity. Both executables receive the same environment.
    env = {**os.environ, 'OMP_PROC_BIND': 'false'}
    with args.output.open('w', newline='') as output, tempfile.TemporaryDirectory(dir=args.work_dir) as temp:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for dataset_arg in args.dataset:
            label, source_text = dataset_arg.split('=', 1)
            if not label or Path(label).name != label or label in ('.', '..'):
                parser.error('dataset label must be a directory basename')
            source = Path(source_text).resolve()
            metadata = json.loads((source/'metadata.json').read_text())
            dimensions = metadata['num_dimensions']
            for side in ['a', 'b']:
                expected = metadata['num_points'][side]*dimensions*4
                if (source/f'{side.upper()}.bin').stat().st_size != expected:
                    parser.error(f'{source}: expected float32 {side.upper()}.bin ({expected} bytes)')
            manifest['inputs'][label] = {'path': str(source), 'metadata': metadata,
                'sha256': {name: digest(source/name) for name in ['A.bin', 'B.bin', 'metadata.json']}}
            fixtures = {}
            for variant in binaries:
                fixture = Path(temp)/label/variant
                fixture.mkdir(parents=True)
                for name in ['A.bin', 'B.bin']:
                    try: os.link(source/name, fixture/name)
                    except OSError: shutil.copyfile(source/name, fixture/name)
                shutil.copyfile(source/'metadata.json', fixture/'metadata.json')
                fixtures[variant] = fixture
            for norm in args.norms:
                for threads in args.threads:
                    for mode in args.modes:
                        reference_weights = None
                        reference_estimate = None
                        for repeat in range(-args.warmups, args.repeats):
                            for variant in (['before', 'after'] if repeat % 2 == 0 else ['after', 'before']):
                                fixture = fixtures[variant]
                                weight_path = fixture/f'l{norm}_qalsh_weights_A.bin'
                                command = [str(binaries[variant]), '-T', str(threads), '--use-fixed-seed',
                                    'estimate', '-p', str(norm), '-d', str(fixture), '--direction', 'ab',
                                    'sampling', '--use-cache']
                                if args.samples: command += ['-n', str(args.samples)]
                                command += ['qalsh', '-c', '2']
                                if mode == 'cold': weight_path.unlink(missing_ok=True)
                                elif not weight_path.exists():
                                    subprocess.run(command, env=env, check=True, capture_output=True, text=True)
                                start = time.perf_counter()
                                result = subprocess.run(command, env=env, capture_output=True, text=True)
                                wall_ms = (time.perf_counter()-start)*1000
                                if result.returncode:
                                    raise RuntimeError(f'{command}\n{result.stdout}\n{result.stderr}')
                                estimate = float(re.search(r'^Estimated Distance: ([^\n]+)', result.stdout, re.M)[1])
                                estimator_ms = float(re.search(r'^Time Consumed: ([0-9.]+)', result.stdout, re.M)[1])
                                weight_hash = digest(weight_path)
                                if reference_weights is None:
                                    reference_weights, reference_estimate = weight_hash, estimate
                                if weight_hash != reference_weights:
                                    raise RuntimeError(f'{label} p={norm}: weight bytes changed')
                                # OpenMP reductions may associate the same per-sample terms differently.
                                tolerance = 0.0 if threads == 1 else 2e-6
                                if not math.isclose(estimate, reference_estimate, rel_tol=tolerance, abs_tol=0.0):
                                    raise RuntimeError(f'{label} p={norm} t={threads}: estimate changed: '
                                                       f'{reference_estimate} -> {estimate}')
                                if repeat < 0: continue
                                row = dict(dataset=label, norm=norm, threads=threads, samples=args.samples,
                                    mode=mode, repeat=repeat, variant=variant, wall_ms=f'{wall_ms:.3f}',
                                    estimator_ms=estimator_ms, estimate=estimate, weight_sha256=weight_hash,
                                    command=json.dumps(command))
                                writer.writerow(row); output.flush(); rows.append(row)
                        case = [row for row in rows if row['dataset']==label and row['norm']==norm and
                                row['threads']==threads and row['mode']==mode]
                        medians = {variant: statistics.median(float(row['wall_ms']) for row in case
                                   if row['variant']==variant) for variant in binaries}
                        print(f'{label} L{norm} T{threads} {mode}: {medians["before"]:.2f} -> '
                              f'{medians["after"]:.2f} ms ({medians["before"]/medians["after"]:.3f}x)', flush=True)
    args.output.with_suffix('.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
