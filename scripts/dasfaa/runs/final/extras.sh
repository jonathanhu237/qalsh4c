#!/usr/bin/env bash
# Phase C: thread check (12 threads), |A| scaling at fixed B, and the c sweep. HNSW weights do not depend on QALSH4C
# and come from the runs of 2026-11-20/21.
set -u
source $(dirname $0)/common.sh
# 1. Thread check on two workloads: QALSH4C at 24 and 12 threads with the same projections (fixed seed), so the
# comparison is not blurred by the projection draw.
OUT=$F/threads.txt
for ds in trevi gist; do for p in 1 2; do
  for threads in 24 12; do
    W=$F/work-thr; fresh $ds $W
    log=$F/logs/thr-$ds-$p-$threads-qalsh4c.log
    $BIN -T $threads --use-fixed-seed --initial-radius $R0 $(qopt $p) estimate -p $p -d $W --direction ab sampling -n $(tval $p) --use-cache qalsh -c 2 > $log 2>&1
    echo "$ds $p $threads qalsh4c-fixedseed $(tc $log) $(grep -o "index_ms=[0-9.]*" $log | cut -d= -f2) $(grep -o "queries_ms=[0-9.]*" $log | cut -d= -f2) $(grep -o "weights_ms=[0-9.]*" $log | cut -d= -f2)" >> $OUT
  done
done; done
rm -rf $F/work-thr
echo "threads done $(date +%H:%M)" >> $F/progress.log
# 2. |A| scaling with B fixed: DBpedia (l2) and Trevi (l1), nested prefixes of A. Exact scan at 12 threads (faster than
# 24 for the exact scan), QALSH4C three times at 24 threads.
mkdir -p $F/scaling/data
OUT=$F/scaling/scaling.txt
scale_dir() {  # dataset, source count -> directory with the first n source points
  local base=${DIR[$1]} n=$2 d=$F/scaling/data/$1-A$2
  if [ ! -d $d ]; then
    mkdir -p $d; dim=$(python3 -c "import json;print(json.load(open('$base/metadata.json'))['num_dimensions'])")
    head -c $((n * dim * 4)) $base/A.bin > $d/A.bin; ln -s $base/B.bin $d/B.bin
    python3 -c "
import json; m=json.load(open('$base/metadata.json')); m['num_points']['a']=$n
for k in m['chamfer_distances']: m['chamfer_distances'][k]={'ab': None, 'ba': None}
json.dump(m, open('$d/metadata.json','w'), indent=4)"
  fi
  echo $d
}
for spec in "dbpedia 2 2000 5000 10000 20000 50000" "trevi 1 2000 5000 10000 20000"; do
  set -- $spec; ds=$1; p=$2; shift 2
  for n in "$@"; do
    d=$(scale_dir $ds $n)
    log=$F/logs/scale-$ds-$n-exact12.log
    $BIN -T 12 estimate -p $p -d $d --direction ab ann linear_scan > $log 2>&1
    echo "$ds $p $n exact12 $(tc $log)" >> $OUT
    for rep in 1 2 3; do
      W=$F/work-scale; rm -rf $W; mkdir -p $W; for f in A.bin B.bin metadata.json; do ln -s $d/$f $W/$f; done
      log=$F/logs/scale-$ds-$n-$rep-qalsh4c.log
      $BIN -T 24 --initial-radius $R0 $(qopt $p) estimate -p $p -d $W --direction ab sampling -n $(tval $p) --use-cache qalsh -c 2 > $log 2>&1
      ix=$(grep -o "index_ms=[0-9.]*" $log | cut -d= -f2); qu=$(grep -o "queries_ms=[0-9.]*" $log | cut -d= -f2)
      wt=$(grep -o "weights_ms=[0-9.]*" $log | cut -d= -f2)
      echo "$ds $p $n qalsh4c $(tc $log) $ix $qu $wt" >> $OUT
    done
  done
done
rm -rf $F/work-scale $F/scaling/data
echo "scaling done $(date +%H:%M)" >> $F/progress.log
# 3. c sweep with beta = 1 (QALSH's rule gives m, l per c), N = 50 per cell; failure_rate.py takes T from the binary's
# certified kappa of the continuous front.
mkdir -p $F/csweep
for ds in trevi enron; do for p in 1 2; do for c in 1.5 3; do
  ex=$(exact_file $ds $p)
  python3 $SCRIPTS/failure_rate.py --binary $BIN --dataset ${DIR[$ds]} --exact $ex --norm $p --runs 50 \
    --eps 0.1 --initial-radius $R0 --approximation-ratio $c \
    --out $F/csweep/${ds}_l${p}_c$c.jsonl >> $F/logs/csweep.log 2>&1 || echo "csweep failed $ds $p $c" >> $F/progress.log
done; done; done
echo "csweep done $(date +%H:%M)" >> $F/progress.log
