#!/usr/bin/env bash
# VLDB phase: thread scaling. Our exact scan once and QALSH4C three times (cold) at 1, 2, 4, 8, 12 and 24 threads,
# both norms, on the given datasets. Output: $V/threads.txt.
set -u
source $(dirname $(readlink -f $0))/common.sh
OUT=$V/threads.txt
[ -s $OUT ] || echo "# dataset norm threads method rep total_ms index_ms queries_ms weights_ms" > $OUT
W=$V/work-threads
for ds in "$@"; do
  cat ${DIR[$ds]}/A.bin ${DIR[$ds]}/B.bin > /dev/null
  for p in 1 2; do
    for t in 1 2 4 8 12 24; do
      grep -q "^$ds $p $t exact " $OUT || {
        log=$V/logs/threads-$ds-$p-$t-exact.log
        $BIN -T $t estimate -p $p -d ${DIR[$ds]} --direction ab ann linear_scan > $log 2>&1
        echo "$ds $p $t exact 1 $(tc $log)" >> $OUT; }
      for rep in 1 2 3; do
        grep -q "^$ds $p $t qalsh4c $rep " $OUT && continue
        fresh $ds $W
        log=$V/logs/threads-$ds-$p-$t-$rep-qalsh4c.log
        $BIN -T $t --initial-radius $R0 estimate -p $p -d $W --direction ab sampling -n $(tval $p) --use-cache \
          qalsh -c 2 > $log 2>&1
        echo "$ds $p $t qalsh4c $rep $(tc $log) $(field index_ms $log) $(field queries_ms $log) $(field weights_ms $log)" \
          >> $OUT
      done
    done
  done
done
rm -rf $W
echo "threads done $* $(date +%H:%M)" >> $V/progress.log
