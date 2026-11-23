#!/usr/bin/env bash
# Phase B: full-randomness test, N = 50 reruns of the whole pipeline per dataset and norm (fresh projections, index
# and crude distances each run), one estimate per tolerance; crude distances saved for the accuracy comparison.
set -u
source $(dirname $0)/common.sh
mkdir -p $F/randomness
for ds in ${@:-$LINEUP}; do
  for p in 1 2; do
    ex=$(exact_file $ds $p)
    python3 $SCRIPTS/failure_rate.py --binary $BIN --dataset ${DIR[$ds]} --exact $ex --norm $p --runs 50 \
      --kappa $(kval $p) --eps 0.3 0.1 --initial-radius $R0 --extra-args "$(qopt $p)" \
      --out $F/randomness/${ds}_l$p.jsonl --weights-dir $F/randomness/weights/${ds}_l$p >> $F/logs/randomness.log 2>&1 \
      || echo "randomness failed $ds $p" >> $F/progress.log
  done
done
echo "randomness done $(date +%H:%M)" >> $F/progress.log
