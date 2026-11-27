#!/usr/bin/env bash
# VLDB phase: total peak resident memory (GNU time, maximum resident set size) of the exact scan and of QALSH4C, once
# per dataset and norm, at the thread counts of the cost phase.
set -u
source $(dirname $(readlink -f $0))/common.sh
OUT=$V/memory.txt
[ -s $OUT ] || echo "# dataset norm method max_rss_kb wall_s" > $OUT
W=$V/work-memory
for ds in "$@"; do
  for p in 1 2; do
    # A run directory may hold the exact rows of an earlier run (the exact scan does not depend on QALSH4C).
    grep -q "^$ds $p exact " $OUT || {
      log=$V/logs/memory-$ds-$p-exact.log
      /usr/bin/time -v $BIN -T 12 estimate -p $p -d ${DIR[$ds]} --direction ab ann linear_scan > $log 2>&1
      echo "$ds $p exact $(grep 'Maximum resident' $log | awk '{print $NF}') $(tc $log)" >> $OUT; }
    grep -q "^$ds $p qalsh4c " $OUT && continue
    fresh $ds $W
    log=$V/logs/memory-$ds-$p-qalsh4c.log
    /usr/bin/time -v $BIN -T 24 --initial-radius $R0 estimate -p $p -d $W --direction ab sampling -n $(tval $p) \
      --use-cache qalsh -c 2 > $log 2>&1
    echo "$ds $p qalsh4c $(grep 'Maximum resident' $log | awk '{print $NF}') $(tc $log)" >> $OUT
  done
done
rm -rf $W
echo "memory done $* $(date +%H:%M)" >> $V/progress.log
