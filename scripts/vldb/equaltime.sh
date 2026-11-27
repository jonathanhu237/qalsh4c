#!/usr/bin/env bash
# VLDB phase: equal-time budgets of uniform sampling and quadtree weights on the datasets added for VLDB, as in part (a)
# of the DASFAA validate phase (scripts/dasfaa/runs/final/validate.sh). Each method is timed with T = 1 (fixed cost)
# and with a count near the budget, the mean QALSH4C time of the cost phase; recalib.py then gives each method the
# largest T that fits in that time and forms its errors with the binary.
set -u
source $(dirname $(readlink -f $0))/common.sh
python3 $SCRIPTS/summarize_cost.py $V --json $V/cost_summary.json > /dev/null
# recalib.py reads the exact distances and the certified T from the results directory.
ln -sfn $F/exact $V/exact
[ -e $V/kappa.txt ] || ln -s $F/kappa.txt $V/kappa.txt
OUT=$V/validate.txt
touch $OUT
BUDGETS=$(python3 - "$@" <<EOF
import json, sys
new = {(r["dataset"], r["norm"]): r for r in json.load(open("$V/cost_summary.json")) if r["threads"] == 24}
quad = {}
for line in open("$V/cost.txt"):
    f = line.split()
    if len(f) >= 9 and f[4] == "quadtree":
        quad[(f[0], int(f[1]))] = float(f[8]) / 1000  # weights_ms of the quadtree run
T = {1: $(tval 1), 2: $(tval 2)}
for ds in sys.argv[1:]:
    for p in (1, 2):
        r = new[(ds, p)]
        per = r["sampling_s"] / T[p]
        tu = max(2, int(r["qalsh4c_s"] / per))
        tq = int((r["qalsh4c_s"] - quad.get((ds, p), 1e9)) / per)
        print(f"{ds}:{p}:{tu}:{tq if tq >= 2 else 0}")
EOF
)
echo "equal-time budgets: $BUDGETS" >> $V/progress.log
for spec in $BUDGETS; do
  IFS=: read -r ds p tu tq <<< "$spec"
  cat ${DIR[$ds]}/A.bin ${DIR[$ds]}/B.bin > /dev/null
  for threads in 12 24; do
    for n in 1 $tu; do
      grep -q "^$ds $p uniform $threads $n " $OUT && continue
      log=$V/logs/val-$ds-$p-uniform-$threads-$n.log
      $BIN -T $threads estimate -p $p -d ${DIR[$ds]} --direction ab sampling -n $n uniform > $log 2>&1
      echo "$ds $p uniform $threads $n $(tc $log)" >> $OUT
    done
  done
  if [ "$tq" != 0 ]; then
    for n in 1 $tq; do
      grep -q "^$ds $p quadtree 24 $n " $OUT && continue
      W=$V/work-val; fresh $ds $W
      log=$V/logs/val-$ds-$p-quadtree-$n.log
      $BIN -T 24 estimate -p $p -d $W --direction ab sampling -n $n --use-cache quadtree --max-levels 3 > $log 2>&1
      echo "$ds $p quadtree 24 $n $(tc $log)" >> $OUT
    done
  fi
done
rm -rf $V/work-val
args=""
for ds in "$@"; do args="$args $ds=${DIR[$ds]}"; done
python3 $SCRIPTS/runs/final/recalib.py $V $BIN $args >> $V/logs/recalib.log 2>&1 || echo "recalib failed" >> $V/progress.log
echo "equaltime done $* $(date +%H:%M)" >> $V/progress.log
