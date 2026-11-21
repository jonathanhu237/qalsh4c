#!/usr/bin/env bash
# Phase A: end-to-end cost on an otherwise idle machine, 24 threads. Our exact scan three times, QALSH4C five times
# (cold: new projections and index each run, so the spread includes the projection draw), one FAISS exact run, and
# uniform sampling and Bakshi et al.'s quadtree weights at the same T.
set -u
source $(dirname $0)/common.sh
OUT=$F/cost.txt
W=$F/work-cost
[ -s $OUT ] || echo "# dataset norm threads rep method total_ms index_ms queries_ms weights_ms D/CH stats" >> $OUT
for ds in ${@:-$LINEUP}; do
  # Warm the page cache so that no timed run reads from disk (GIST lives on a hard disk).
  cat ${DIR[$ds]}/A.bin ${DIR[$ds]}/B.bin > /dev/null
  for p in 1 2; do
    ex=$(exact_file $ds $p)
    for rep in 1 2 3; do
      log=$F/logs/cost-$ds-$p-24-$rep-exact.log
      $BIN -T 24 estimate -p $p -d ${DIR[$ds]} --direction ab ann linear_scan > $log 2>&1
      echo "$ds $p 24 $rep exact $(tc $log)" >> $OUT
    done
    for rep in 1 2 3 4 5; do
      fresh $ds $W
      log=$F/logs/cost-$ds-$p-24-$rep-qalsh4c.log
      $BIN -T 24 --initial-radius $R0 $(qopt $p) estimate -p $p -d $W --direction ab sampling -n $(tval $p) --use-cache qalsh -c 2 > $log 2>&1
      ix=$(grep -o "index_ms=[0-9.]*" $log | cut -d= -f2); qu=$(grep -o "queries_ms=[0-9.]*" $log | cut -d= -f2)
      wt=$(grep -o "weights_ms=[0-9.]*" $log | cut -d= -f2); st=$(grep "QALSH stats" $log | sed "s/.*QALSH stats: //")
      dch=$(python3 -c "import numpy as np; w=np.fromfile('$W/l${p}_qalsh_weights_A.bin',dtype=np.float32).astype(np.float64); e=np.fromfile('$ex',dtype=np.float64); print('%.4f' % (w.sum()/e.sum()))")
      echo "$ds $p 24 $rep qalsh4c $(tc $log) $ix $qu $wt $dch [$st]" >> $OUT
    done
    $PY $SCRIPTS/faiss_exact.py --dataset ${DIR[$ds]} --norm $p --threads 24 --repeats 1 --exact $ex >> $F/faiss.jsonl 2>> $F/logs/faiss.err
    log=$F/logs/cost-$ds-$p-uniform.log
    $BIN -T 24 estimate -p $p -d ${DIR[$ds]} --direction ab sampling -n $(tval $p) uniform > $log 2>&1
    echo "$ds $p 24 1 uniform $(tc $log)" >> $OUT
    fresh $ds $W
    log=$F/logs/cost-$ds-$p-quadtree.log
    $BIN -T 24 estimate -p $p -d $W --direction ab sampling -n $(tval $p) --use-cache quadtree --max-levels 3 > $log 2>&1
    wt=$(grep -o "weights_ms=[0-9.]*" $log | cut -d= -f2)
    echo "$ds $p 24 1 quadtree $(tc $log) - - $wt" >> $OUT
    mkdir -p $F/quadtree; cp $W/l${p}_quadtree_l3_weights_A.bin $F/quadtree/${ds}_l$p.bin 2>/dev/null
  done
done
rm -rf $W
echo "cost done $(date +%H:%M)" >> $F/progress.log
