#!/usr/bin/env bash
# Adds cold QALSH4C runs to cost.txt until every dataset and norm has $QALSH_RUNS of them, with the same command as
# cost.sh. Used when cost.sh ran with fewer runs (2026-11-24: 5 runs, then raised to 20 for the mean).
set -u
source $(dirname $0)/common.sh
OUT=$F/cost.txt
W=$F/work-topup
for ds in ${@:-$LINEUP mnist federalist}; do
  cat ${DIR[$ds]}/A.bin ${DIR[$ds]}/B.bin > /dev/null
  for p in 1 2; do
    ex=$(exact_file $ds $p)
    have=$(awk -v d=$ds -v n=$p '$1 == d && $2 == n && $5 == "qalsh4c"' $OUT | wc -l)
    for rep in $(seq $((have + 1)) $QALSH_RUNS); do
      fresh $ds $W
      log=$F/logs/cost-$ds-$p-24-$rep-qalsh4c.log
      $BIN -T 24 --initial-radius $R0 $(qopt $p) estimate -p $p -d $W --direction ab sampling -n $(tval $p) --use-cache qalsh -c 2 > $log 2>&1
      ix=$(grep -o "index_ms=[0-9.]*" $log | cut -d= -f2); qu=$(grep -o "queries_ms=[0-9.]*" $log | cut -d= -f2)
      wt=$(grep -o "weights_ms=[0-9.]*" $log | cut -d= -f2); st=$(grep "QALSH stats" $log | sed "s/.*QALSH stats: //")
      mem=$(grep "Memory Usage" $log | sed "s/.*Memory Usage: //; s/ MB//")
      echo "$ds $p 24 $rep qalsh4c $(tc $log) $ix $qu $wt $(dch $W/l${p}_qalsh_weights_A.bin $ex) [$st memory_mb=$mem]" >> $OUT
    done
  done
done
rm -rf $W
echo "topup done $(date +%H:%M)" >> $F/progress.log
