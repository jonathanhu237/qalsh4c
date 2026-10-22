#!/usr/bin/env python3
"""Split a qalsh-h float32 MNIST base into canonical runtime benchmark inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='Directory containing base.bin and qalsh-h metadata.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-points-a', type=int, default=3000)
    args = parser.parse_args()
    metadata = json.loads((args.source / 'metadata.json').read_text())
    dimensions, count = metadata['num_dimensions'], metadata['num_points_in_base']
    if not 0 < args.num_points_a < count or dimensions < 1:
        parser.error('the split must contain two nonempty point sets')
    source = args.source / 'base.bin'
    if source.stat().st_size != count * dimensions * 4:
        parser.error('base.bin must contain exactly the declared float32 coordinates')
    # Refuse an existing destination so a benchmark never overwrites source data.
    args.output.mkdir(parents=True, exist_ok=False)
    with source.open('rb') as stream:
        remaining = args.num_points_a * dimensions * 4
        with (args.output / 'A.bin').open('wb') as output:
            while remaining:
                block = stream.read(min(remaining, 1024 * 1024))
                if not block:
                    raise RuntimeError('base.bin was truncated during the split')
                output.write(block)
                remaining -= len(block)
        with (args.output / 'B.bin').open('wb') as output:
            shutil.copyfileobj(stream, output)
    canonical = {
        'num_dimensions': dimensions,
        'num_points': {'a': args.num_points_a, 'b': count - args.num_points_a},
        'chamfer_distances': {norm: {'ab': None, 'ba': None} for norm in ['l1', 'l2']},
    }
    (args.output / 'metadata.json').write_text(json.dumps(canonical, indent=2))
    for name in ['A.bin', 'B.bin', 'metadata.json']:
        with (args.output / name).open('rb') as source_file:
            print(name, hashlib.file_digest(source_file, 'sha256').hexdigest())


if __name__ == '__main__':
    main()
