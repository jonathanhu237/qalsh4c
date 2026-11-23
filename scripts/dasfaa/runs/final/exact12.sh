#!/usr/bin/env bash
# Exact baselines at 12 threads (one per physical core), which the thread check found faster than 24 for the exact
# scan: our scan three times and FAISS once per dataset and norm, plus the |A|-scaling exact runs. Waits for the
# main chain to finish so that nothing else shares the CPU.
set -u
source $(dirname $0)/common.sh
until grep -q "ALL DONE" $F/progress.log 2>/dev/null; do sleep 30; done
OUT=$F/cost.txt
for ds in $LINEUP mnist federalist; do
  cat ${DIR[$ds]}/A.bin ${DIR[$ds]}/B.bin > /dev/null
  for p in 1 2; do
    ex=$(exact_file $ds $p)
    for rep in 1 2 3; do
      log=$F/logs/cost-$ds-$p-12-$rep-exact.log
      $BIN -T 12 estimate -p $p -d ${DIR[$ds]} --direction ab ann linear_scan > $log 2>&1
      echo "$ds $p 12 $rep exact $(tc $log)" >> $OUT
    done
    $PY $SCRIPTS/faiss_exact.py --dataset ${DIR[$ds]} --norm $p --threads 12 --repeats 1 --exact $ex >> $F/faiss.jsonl 2>> $F/logs/faiss.err
  done
done
echo "exact12 done $(date +%H:%M)" >> $F/progress.log
OUT=$F/scaling/scaling.txt
for spec in "dbpedia 2 2000 5000 10000 20000 50000" "trevi 1 2000 5000 10000 20000"; do
  set -- $spec; ds=$1; p=$2; shift 2
  for n in "$@"; do
    d=$F/scaling/data/$ds-A$n
    log=$F/logs/scale12-$ds-$n-exact.log
    $BIN -T 12 estimate -p $p -d $d --direction ab ann linear_scan > $log 2>&1
    echo "$ds $p $n exact12 $(tc $log)" >> $OUT
  done
done
echo "scaling12 done $(date +%H:%M)" >> $F/progress.log
