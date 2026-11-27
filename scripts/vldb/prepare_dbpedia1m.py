"""Build the full DBpedia workloads from Supabase/dbpedia-openai-3-large-1M (OpenAI text-embedding-3-large, 3072-d).

The 26 parquet files hold about 1M entities with a list<double> embedding each. The vectors are converted to float32,
duplicates of _id are dropped, and the collection is shuffled once with a fixed seed. Two datasets are written in the
binary's layout (A.bin, B.bin, metadata.json):
  dbpedia1m        A = the first 50,000 vectors, B = all the others (the large dataset that replaces GIST),
  dbpedia1m-scale  A = the first 500,000 vectors, B = the last 500,000 (disjoint; nested prefixes of both give the
                   two-sided scaling series).

Usage: python prepare_dbpedia1m.py RAW_DIR OUT_ROOT
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

DIM = 3072


def write(out: str, a: np.ndarray, b: np.ndarray) -> None:
    os.makedirs(out, exist_ok=True)
    a.tofile(os.path.join(out, "A.bin"))
    b.tofile(os.path.join(out, "B.bin"))
    meta = {"num_dimensions": DIM, "num_points": {"a": int(a.shape[0]), "b": int(b.shape[0])},
            "chamfer_distances": {"l1": {"ab": None, "ba": None}, "l2": {"ab": None, "ba": None}}}
    with open(os.path.join(out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=4)


def main() -> None:
    raw, root = sys.argv[1], sys.argv[2]
    files = sorted(glob.glob(os.path.join(raw, "dbpedia_openai_3_large_*.parquet")))
    if len(files) != 26:
        raise SystemExit(f"expected 26 parquet files, found {len(files)}")
    ids: list[str] = []
    parts: list[np.ndarray] = []
    for path in files:
        table = pq.read_table(path, columns=["_id", "embedding"])
        ids.extend(table.column("_id").to_pylist())
        emb = table.column("embedding").combine_chunks()
        values = emb.values.to_numpy(zero_copy_only=False)
        parts.append(values.reshape(-1, DIM).astype(np.float32))
        print(f"{os.path.basename(path)}: {parts[-1].shape[0]} rows", flush=True)
    x = np.concatenate(parts)
    del parts
    _, first = np.unique(np.array(ids), return_index=True)
    keep = np.sort(first)
    print(f"rows {x.shape[0]}, distinct _id {keep.size}", flush=True)
    x = x[keep]
    rng = np.random.default_rng(20261124)
    x = x[rng.permutation(x.shape[0])]
    norms = np.linalg.norm(x, axis=1)
    print(f"norms min {norms.min():.4f} max {norms.max():.4f}", flush=True)
    write(os.path.join(root, "dbpedia1m"), x[:50000], x[50000:])
    half = 500000
    write(os.path.join(root, "dbpedia1m-scale"), x[:half], x[x.shape[0] - half:])
    print("done", flush=True)


if __name__ == "__main__":
    main()
