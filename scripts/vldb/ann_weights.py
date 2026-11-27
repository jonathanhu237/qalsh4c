"""Crude distances from in-memory ANN indexes (CrudeNN baselines without a computable kappa), l2 only.

For every source vector a, the index returns one target b~ and the crude distance is the exact distance ||a - b~||_2,
recomputed in float64 from the original vectors. It is the distance to an actual target vector, so D_a >= d(a, B),
and the weights can feed the same importance-sampling estimator (ANN-IS) or be summed (ANN-Sum). An empty answer
(possible with IVF and few probes) falls back to the first target vector, which is still an upper bound.

Methods and the parameters swept at query time (the index is built once per build setting):
  hnsw     hnswlib, --m and --ef-construction; query ef
  ivfflat  FAISS IndexIVFFlat with nlist lists, trained on a fixed sample; query nprobe
  ivfpq    FAISS IndexIVFPQ (nlist lists, PQ with --pq-m sub-quantizers of 8 bits) wrapped in IndexRefineFlat, which
           re-ranks the r best PQ candidates by exact distance; query nprobe and r

Usage: python ann_weights.py --dataset DIR --out-dir OUT --method hnsw --m 32 --ef-construction 200 --query 10 40 100
Writes OUT/<tag>.bin (float32 weights, one per source vector) and appends one JSON line per query setting to
OUT/timing.jsonl with the build, query and distance times in seconds.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def load(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    with open(os.path.join(dataset, "metadata.json")) as f:
        meta = json.load(f)
    d = meta["num_dimensions"]
    a = np.fromfile(os.path.join(dataset, "A.bin"), dtype=np.float32).reshape(meta["num_points"]["a"], d)
    b = np.fromfile(os.path.join(dataset, "B.bin"), dtype=np.float32).reshape(meta["num_points"]["b"], d)
    return a, b


def exact_distances(a: np.ndarray, b: np.ndarray, ids: np.ndarray) -> np.ndarray:
    ids = np.where(ids < 0, 0, ids)
    out = np.empty(a.shape[0], dtype=np.float32)
    for s in range(0, a.shape[0], 8192):
        diff = a[s:s + 8192].astype(np.float64) - b[ids[s:s + 8192]].astype(np.float64)
        out[s:s + 8192] = np.sqrt((diff * diff).sum(axis=1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--method", choices=("hnsw", "ivfflat", "ivfpq"), required=True)
    parser.add_argument("--m", type=int, default=32, help="hnsw: graph degree M")
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--pq-m", type=int, default=64, help="ivfpq: number of 8-bit sub-quantizers")
    parser.add_argument("--train-size", type=int, default=65536)
    parser.add_argument("--query", type=int, nargs="+", required=True, help="ef (hnsw) or nprobe (ivf)")
    parser.add_argument("--refine", type=int, nargs="+", default=[1], help="ivfpq: exact re-ranking depth r")
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    start = time.perf_counter()
    a, b = load(args.dataset)
    load_s = time.perf_counter() - start
    rng = np.random.default_rng(args.seed)

    start = time.perf_counter()
    if args.method == "hnsw":
        import hnswlib
        index = hnswlib.Index(space="l2", dim=a.shape[1])
        index.set_num_threads(args.threads)
        index.init_index(max_elements=b.shape[0], ef_construction=args.ef_construction, M=args.m,
                         random_seed=args.seed)
        index.add_items(b, np.arange(b.shape[0]))
        build = f"M{args.m}-efC{args.ef_construction}"
    else:
        import faiss
        faiss.omp_set_num_threads(args.threads)
        quantizer = faiss.IndexFlatL2(a.shape[1])
        if args.method == "ivfflat":
            index = faiss.IndexIVFFlat(quantizer, a.shape[1], args.nlist)
            build = f"nlist{args.nlist}"
        else:
            base = faiss.IndexIVFPQ(quantizer, a.shape[1], args.nlist, args.pq_m, 8)
            index = faiss.IndexRefineFlat(base)
            build = f"nlist{args.nlist}-pq{args.pq_m}"
        sample = b[rng.choice(b.shape[0], min(args.train_size, b.shape[0]), replace=False)]
        index.train(sample)
        index.add(b)
    build_s = time.perf_counter() - start

    settings = [(q, r) for q in args.query for r in (args.refine if args.method == "ivfpq" else [None])]
    for q, r in settings:
        if args.method == "hnsw":
            index.set_ef(q)
            start = time.perf_counter()
            ids, _ = index.knn_query(a, k=1)
        else:
            import faiss
            if args.method == "ivfflat":
                index.nprobe = q
            else:
                faiss.extract_index_ivf(index.base_index).nprobe = q
                index.k_factor = r
            start = time.perf_counter()
            _, ids = index.search(a, 1)
        query_s = time.perf_counter() - start
        ids = ids[:, 0].astype(np.int64)
        start = time.perf_counter()
        weights = exact_distances(a, b, ids)
        dist_s = time.perf_counter() - start
        tag = f"{args.method}-{build}-q{q}" + (f"-r{r}" if r is not None else "")
        weights.tofile(os.path.join(args.out_dir, f"{tag}.bin"))
        record = {"tag": tag, "method": args.method, "build": build, "query_param": q, "refine": r,
                  "load_s": load_s, "build_s": build_s, "query_s": query_s, "dist_s": dist_s,
                  "empty": int((ids < 0).sum())}
        with open(os.path.join(args.out_dir, "timing.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
