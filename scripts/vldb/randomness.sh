#!/usr/bin/env bash
# VLDB phase: the full-randomness test of the DASFAA runs on the new datasets. N = 50 reruns of the whole pipeline per
# dataset and norm (fresh projections each run), estimates at eps 0.3 and 0.1; crude distances saved per run.
set -u
source $(dirname $(readlink -f $0))/common.sh
mkdir -p $V/randomness
for ds in "$@"; do
  for p in 1 2; do
    ex=$(exact_file $ds $p)
    python3 $SCRIPTS/failure_rate.py --binary $BIN --dataset ${DIR[$ds]} --exact $ex --norm $p --runs 50 \
      --eps 0.3 0.1 --initial-radius $R0 --out $V/randomness/${ds}_l$p.jsonl \
      --weights-dir $V/randomness/weights/${ds}_l$p >> $V/logs/randomness.log 2>&1 \
      || echo "randomness failed $ds $p" >> $V/progress.log
  done
done
echo "randomness done $* $(date +%H:%M)" >> $V/progress.log
