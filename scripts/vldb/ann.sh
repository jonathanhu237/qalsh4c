#!/usr/bin/env bash
# VLDB phase: CrudeNN choices on each dataset (l2): HNSW (two builds), IVF-Flat (two list counts) and IVF-PQ with
# exact re-ranking, summed or used for importance sampling at the certified T of QALSH4C at eps = 0.2, 0.1 and 0.05,
# against uniform sampling and quadtree weights over a grid of T, and QALSH4C (crude distances saved by all 50
# randomness runs).
set -u
source $(dirname $(readlink -f $0))/common.sh
TS="$(tval 2 main 2 0.2) $(tval 2 main 2 0.1) $(tval 2 main 2 0.05)"
for ds in "$@"; do
  qw=$V/randomness/weights/${ds}_l2; [ -d $qw ] || qw=$F/randomness/weights/${ds}_l2
  # The ANN builds and queries do not depend on our binary: reuse those of the first VLDB run when they exist.
  old=/home/jonathanhu237/qalsh4c-results/vldb-20261124/ann/weights/$ds
  reuse=""; [ "$V" != "${old%/ann/weights/$ds}" ] && [ -s $old/timing.jsonl ] && reuse="--reuse-weights $old"
  bound=""; [ -n "${VLDB_BIN:-}" ] && bound="--upper-bounds"
  quad=""; [ -s $V/quadtree/${ds}_l2.bin ] && quad="--quadtree-weights $V/quadtree/${ds}_l2.bin"
  # IVF-PQ needs the number of sub-quantizers to divide d (5,408 on P53, 1,369 on Enron).
  pqm=64; [ $ds = p53 ] && pqm=52; [ $ds = enron ] && pqm=37
  cat ${DIR[$ds]}/A.bin ${DIR[$ds]}/B.bin > /dev/null
  python3 $(dirname $(readlink -f $0))/anncmp.py --name $ds --dataset ${DIR[$ds]} --exact $(exact_file $ds 2) \
    --binary $BIN --venv $VENV --out $V/ann --qalsh-weights $qw --qalsh-runs 50 --repeats 400 --T $TS \
    --uniform-T 16 64 256 1024 4096 16384 $TS $reuse $bound $quad \
    --ann "--method hnsw --m 16 --ef-construction 100 --query 10 40 100" \
    --ann "--method hnsw --m 32 --ef-construction 200 --query 10 40 100" \
    --ann "--method ivfflat --nlist 256 --query 1 4 16 64" \
    --ann "--method ivfflat --nlist 1024 --query 1 4 16 64" \
    --ann "--method ivfpq --nlist 1024 --pq-m $pqm --query 4 16 64 --refine 1 32" >> $V/logs/ann-$ds.log 2>&1 \
    || echo "ann failed $ds" >> $V/progress.log
  echo "ann done $ds $(date +%H:%M)" >> $V/progress.log
done
