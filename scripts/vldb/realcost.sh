#!/usr/bin/env bash
# VLDB phase: end-to-end cost on the new datasets, as in the DASFAA cost phase (scripts/dasfaa/runs/final/cost.sh).
# Our exact scan three times at 12 and 24 threads, FAISS exact (l2) three times at 12 and 24 threads, QALSH4C 20 times
# cold at 24 threads, uniform sampling and Bakshi et al.'s quadtree weights at the same T.
set -u
source $(dirname $(readlink -f $0))/common.sh
OUT=$V/cost.txt
[ -s $OUT ] || echo "# dataset norm threads rep method total_ms index_ms queries_ms weights_ms D/C stats" > $OUT
W=$V/work-cost
for ds in "$@"; do
  cat ${DIR[$ds]}/A.bin ${DIR[$ds]}/B.bin > /dev/null
  for p in 1 2; do
    ex=$(exact_file $ds $p)
    for threads in 12 24; do
      for rep in 1 2 3; do
        log=$V/logs/cost-$ds-$p-$threads-$rep-exact.log
        $BIN -T $threads estimate -p $p -d ${DIR[$ds]} --direction ab ann linear_scan > $log 2>&1
        echo "$ds $p $threads $rep exact $(tc $log)" >> $OUT
      done
      if [ $p = 2 ]; then
        $VENV $SCRIPTS/faiss_exact.py --dataset ${DIR[$ds]} --norm 2 --threads $threads --repeats 3 --exact $ex \
          | while read -r line; do echo "{\"dataset\": \"$ds\", \"norm\": 2, \"threads\": $threads, ${line#\{}"; done \
          >> $V/faiss.jsonl
      fi
    done
    for rep in $(seq 1 $QALSH_RUNS); do
      fresh $ds $W
      log=$V/logs/cost-$ds-$p-24-$rep-qalsh4c.log
      $BIN -T 24 --initial-radius $R0 estimate -p $p -d $W --direction ab sampling -n $(tval $p) --use-cache qalsh \
        -c 2 > $log 2>&1
      st=$(grep "QALSH stats" $log | sed "s/.*QALSH stats: //")
      echo "$ds $p 24 $rep qalsh4c $(tc $log) $(field index_ms $log) $(field queries_ms $log) $(field weights_ms $log)" \
           "$(dch $W/l${p}_qalsh_weights_A.bin $ex) [$st]" >> $OUT
    done
    log=$V/logs/cost-$ds-$p-uniform.log
    $BIN -T 24 estimate -p $p -d ${DIR[$ds]} --direction ab sampling -n $(tval $p) uniform > $log 2>&1
    echo "$ds $p 24 1 uniform $(tc $log)" >> $OUT
    fresh $ds $W
    log=$V/logs/cost-$ds-$p-quadtree.log
    $BIN -T 24 estimate -p $p -d $W --direction ab sampling -n $(tval $p) --use-cache quadtree --max-levels 3 > $log 2>&1
    echo "$ds $p 24 1 quadtree $(tc $log) - - $(field weights_ms $log)" >> $OUT
    mkdir -p $V/quadtree; cp $W/l${p}_quadtree_l3_weights_A.bin $V/quadtree/${ds}_l$p.bin 2>/dev/null
  done
done
rm -rf $W
echo "realcost done $* $(date +%H:%M)" >> $V/progress.log
