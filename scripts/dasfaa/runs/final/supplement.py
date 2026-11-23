"""Optional supplement suggested by GPT r3: (1) do uniform sampling and quadtree weights meet the tolerance at their
equal-time sample counts? Failure counts over many draws. (2) spread of QALSH4C's index + crude time over the 50
full-randomness runs. Usage: python3 supplement.py FINAL_DIR"""
import glob, json, sys
import numpy as np

final = sys.argv[1]
rng = np.random.default_rng(20261123)
timed = json.load(open(f"{final}/time_accuracy_timed.json"))
DRAWS = 2000

def err_is(w, exact, t, draws):
    ch = exact.sum(); tot = w.sum(); prob = w / tot; sup = np.flatnonzero(prob > 0)
    out = []
    for start in range(0, draws, 200):
        picks = rng.choice(sup, size=(min(200, draws - start), t), replace=True, p=prob[sup])
        out.append(np.abs(np.mean(tot / w[picks] * exact[picks], axis=1) - ch) / ch)
    return np.concatenate(out)

def err_uniform(exact, t, draws):
    ch = exact.sum(); out = []
    for start in range(0, draws, 200):
        picks = rng.integers(0, exact.size, size=(min(200, draws - start), t))
        out.append(np.abs(exact[picks].mean(axis=1) * exact.size - ch) / ch)
    return np.concatenate(out)

print("cell          | QALSH4C 50 runs: fails/max | uniform (T): fails/2000, max | quadtree (T): fails/2000, max | QALSH4C index+crude s: min/median/max over 50 runs")
for key, row in sorted(timed.items()):
    ds, p = key.rsplit("_l", 1)
    exact = np.fromfile(f"{final}/exact/{key}.bin", dtype=np.float64)
    runs = [json.loads(l) for l in open(f"{final}/randomness/{key}.jsonl")]
    e01 = np.array([r["errors"]["0.1"]["rel_error"] for r in runs])
    crude = np.array([(r["index_ms"] + r["queries_ms"]) / 1000 for r in runs])
    cells = [f"{key:13s}", f"{int((e01 > 0.1).sum())}/50 max {100 * e01.max():.2f}%"]
    u = row.get("uniform")
    if u:
        e = err_uniform(exact, u["T"], DRAWS)
        cells.append(f"T={u['T']}: {int((e > 0.1).sum())} max {100 * e.max():.2f}%")
    q = row.get("quadtree")
    if q:
        w = np.fromfile(f"{final}/quadtree/{key}.bin", dtype=np.float32).astype(np.float64)
        e = err_is(w, exact, q["T"], DRAWS)
        cells.append(f"T={q['T']}: {int((e > 0.1).sum())} max {100 * e.max():.2f}%")
    else:
        cells.append("quadtree: weights alone exceed the budget")
    cells.append(f"{crude.min():.2f}/{np.median(crude):.2f}/{crude.max():.2f}")
    print(" | ".join(cells), flush=True)
