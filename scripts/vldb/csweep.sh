#!/usr/bin/env bash
# VLDB phase: sensitivity to the approximation ratio c, which sets w, m and l (Eq. qalsh-m-l), with the default
# stopping ratio gamma = c^1.5 for each c and the T it certifies at eps = delta = 0.1. Three cold runs per cell at 24
# threads with the seeds 2001 to 2003, on every dataset and norm. Output: $V/csweep.txt.
set -u
source $(dirname $(readlink -f $0))/common.sh
OUT=$V/csweep.txt
[ -s $OUT ] || echo "# dataset norm c rep T total_ms index_ms queries_ms weights_ms D/C stats" > $OUT
W=$V/work-csweep
for ds in "$@"; do
  cat ${DIR[$ds]}/A.bin ${DIR[$ds]}/B.bin > /dev/null
  for p in 1 2; do
    ex=$(exact_file $ds $p)
    for c in 1.5 2 3; do
      T=$(tval $p main $c)
      for rep in 1 2 3; do
        grep -q "^$ds $p $c $rep " $OUT && continue
        fresh $ds $W
        log=$V/logs/csweep-$ds-$p-$c-$rep.log
        $BIN -T 24 --seed $((2000 + rep)) estimate -p $p -d $W --direction ab sampling -n $T --use-cache qalsh \
          -c $c > $log 2>&1
        st=$(grep "QALSH stats" $log | sed "s/.*QALSH stats: //")
        echo "$ds $p $c $rep $T $(tc $log) $(field index_ms $log) $(field queries_ms $log)" \
             "$(field weights_ms $log) $(dch $W/l${p}_qalsh_weights_A.bin $ex) [$st]" >> $OUT
      done
    done
  done
done
rm -rf $W
echo "csweep done $* $(date +%H:%M)" >> $V/progress.log
