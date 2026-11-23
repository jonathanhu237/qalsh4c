#!/usr/bin/env bash
# (a) Time uniform and quadtree sampling with T = 1 (fixed cost) and with a count near the equal-time budget, so that
#     recalib.py can fit the per-draw cost and give each method the largest T within QALSH4C's mean time.
# (b) End-to-end totals of three stopping rules with their own certified T (same projections): the continuous front,
#     the round rule with the end-of-round test (both gamma = c) and QALSH's test on every candidate (gamma = c^2).
# (c) End-to-end totals for c = 1.5 and 3 with their own T, three runs each with new projections.
set -u
source $(dirname $0)/common.sh
OUT=$F/validate.txt
: > $OUT
# dataset:norm:T_uniform:T_quadtree near the budget, from the per-draw cost of QALSH4C's sampling stage.
BUDGETS=$(python3 - <<EOF
import json
new = {(r["dataset"], r["norm"]): r for r in json.load(open("$F/cost_summary.json")) if r["threads"] == 24}
quad = {}
for line in open("$F/cost.txt"):
    f = line.split()
    if len(f) >= 9 and f[4] == "quadtree":
        quad[(f[0], int(f[1]))] = float(f[8]) / 1000  # weights_ms of the quadtree run
T = {1: $(tval 1), 2: $(tval 2)}
for ds in "$LINEUP".split():
    for p in (1, 2):
        r = new[(ds, p)]
        per = r["sampling_s"] / T[p]
        tu = max(2, int(r["qalsh4c_s"] / per))
        tq = int((r["qalsh4c_s"] - quad.get((ds, p), 1e9)) / per)
        print(f"{ds}:{p}:{tu}:{tq if tq >= 2 else 0}")
EOF
)
echo "budgets: $BUDGETS" >> $F/progress.log
for spec in $BUDGETS; do
  IFS=: read -r ds p tu tq <<< "$spec"
  cat ${DIR[$ds]}/A.bin ${DIR[$ds]}/B.bin > /dev/null
  for threads in 12 24; do
    for n in 1 $tu; do
      log=$F/logs/val-$ds-$p-uniform-$threads-$n.log
      $BIN -T $threads estimate -p $p -d ${DIR[$ds]} --direction ab sampling -n $n uniform > $log 2>&1
      echo "$ds $p uniform $threads $n $(tc $log)" >> $OUT
    done
  done
  if [ "$tq" != 0 ]; then
    for n in 1 $tq; do
      W=$F/work-val; fresh $ds $W
      log=$F/logs/val-$ds-$p-quadtree-$n.log
      $BIN -T 24 estimate -p $p -d $W --direction ab sampling -n $n --use-cache quadtree --max-levels 3 > $log 2>&1
      echo "$ds $p quadtree 24 $n $(tc $log)" >> $OUT
    done
  fi
done
echo "equal-time timing done $(date +%H:%M)" >> $F/progress.log
for ds in trevi gist enron dbpedia; do
  cat ${DIR[$ds]}/A.bin ${DIR[$ds]}/B.bin > /dev/null
  for p in 1 2; do
    ex=$(exact_file $ds $p)
    for rule in continuous roundend within; do
      case $rule in
        continuous) ro=""; t=$(tval $p) ;;
        roundend) ro="$ROUNDEND"; t=$(tval $p roundend) ;;
        within) ro="$WITHIN"; t=$(tval $p within) ;;
      esac
      W=$F/work-val; fresh $ds $W
      log=$F/logs/val-stop-$ds-$p-$rule.log
      $BIN -T 24 --use-fixed-seed --initial-radius $R0 $ro estimate -p $p -d $W --direction ab sampling -n $t --use-cache qalsh -c 2 > $log 2>&1
      st=$(grep "QALSH stats" $log | grep -o "hits_per_query=[0-9.]*\|evaluated_per_query=[0-9.]*" | tr "\n" " ")
      echo "$ds $p stop-$rule T=$t $(tc $log) $(grep -o "index_ms=[0-9.]*" $log | cut -d= -f2) $(grep -o "queries_ms=[0-9.]*" $log | cut -d= -f2) $(grep -o "weights_ms=[0-9.]*" $log | cut -d= -f2) $(dch $W/l${p}_qalsh_weights_A.bin $ex) $st" >> $OUT
    done
  done
done
echo "stop totals done $(date +%H:%M)" >> $F/progress.log
for ds in trevi enron; do for p in 1 2; do for c in 1.5 3; do
  t=$(tval $p main $c)
  for rep in 1 2 3; do
    W=$F/work-val; fresh $ds $W
    log=$F/logs/val-c-$ds-$p-$c-$rep.log
    $BIN -T 24 --initial-radius $R0 estimate -p $p -d $W --direction ab sampling -n $t --use-cache qalsh -c $c > $log 2>&1
    echo "$ds $p c=$c T=$t rep=$rep $(tc $log) $(grep -o "index_ms=[0-9.]*" $log | cut -d= -f2) $(grep -o "queries_ms=[0-9.]*" $log | cut -d= -f2) $(grep -o "weights_ms=[0-9.]*" $log | cut -d= -f2)" >> $OUT
  done
done; done; done
rm -rf $F/work-val
echo "validate done $(date +%H:%M)" >> $F/progress.log
